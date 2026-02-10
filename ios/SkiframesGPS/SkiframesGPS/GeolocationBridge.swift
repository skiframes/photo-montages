import SwiftUI
import WebKit
import Combine

// MARK: - Geolocation Mock JavaScript

/// Injected at document start to replace navigator.geolocation with a mock
/// that receives data from Swift via evaluateJavaScript.
/// Falls back to real browser geolocation until Bad Elf data arrives.
private let geolocationMockJS = """
(function() {
    'use strict';

    // Store reference to real geolocation for fallback
    var realGeo = navigator.geolocation;

    var mock = {
        _watchCallbacks: {},
        _nextWatchId: 1,
        _lastPosition: null,
        _active: false,
        _fallbackWatchId: null,

        // Called by Swift via evaluateJavaScript
        _pushPosition: function(data) {
            this._active = true;

            // Stop fallback if it was running
            if (this._fallbackWatchId !== null && realGeo) {
                realGeo.clearWatch(this._fallbackWatchId);
                this._fallbackWatchId = null;
            }

            var position = {
                coords: {
                    latitude: data.latitude,
                    longitude: data.longitude,
                    altitude: data.altitude,
                    accuracy: data.accuracy,
                    altitudeAccuracy: null,
                    heading: null,
                    speed: null,
                    fixType: data.fixType
                },
                timestamp: data.timestamp
            };

            this._lastPosition = position;

            for (var id in this._watchCallbacks) {
                var cb = this._watchCallbacks[id];
                try { cb.success(position); } catch(e) {
                    console.error('GPS callback error:', e);
                }
            }
        },

        watchPosition: function(success, error, options) {
            var id = this._nextWatchId++;
            this._watchCallbacks[id] = { success: success, error: error, options: options };

            // If we already have data, invoke immediately
            if (this._lastPosition) {
                try { success(this._lastPosition); } catch(e) {}
            }

            // Fallback to real geolocation if Bad Elf not active yet
            if (!this._active && realGeo && this._fallbackWatchId === null) {
                var self = this;
                this._fallbackWatchId = realGeo.watchPosition(
                    function(pos) {
                        if (!self._active) {
                            var fbPos = {
                                coords: {
                                    latitude: pos.coords.latitude,
                                    longitude: pos.coords.longitude,
                                    altitude: pos.coords.altitude,
                                    accuracy: pos.coords.accuracy,
                                    altitudeAccuracy: pos.coords.altitudeAccuracy,
                                    heading: pos.coords.heading,
                                    speed: pos.coords.speed,
                                    fixType: 1
                                },
                                timestamp: pos.timestamp
                            };
                            for (var wid in self._watchCallbacks) {
                                try { self._watchCallbacks[wid].success(fbPos); } catch(e) {}
                            }
                        }
                    },
                    function(err) {
                        if (!self._active) {
                            for (var wid in self._watchCallbacks) {
                                var cb = self._watchCallbacks[wid];
                                if (cb.error) {
                                    try { cb.error(err); } catch(e) {}
                                }
                            }
                        }
                    },
                    { enableHighAccuracy: true, maximumAge: 2000, timeout: 10000 }
                );
            }

            return id;
        },

        clearWatch: function(id) {
            delete this._watchCallbacks[id];
            if (Object.keys(this._watchCallbacks).length === 0
                && this._fallbackWatchId !== null && realGeo) {
                realGeo.clearWatch(this._fallbackWatchId);
                this._fallbackWatchId = null;
            }
        },

        getCurrentPosition: function(success, error, options) {
            if (this._lastPosition) {
                success(this._lastPosition);
            } else if (realGeo) {
                realGeo.getCurrentPosition(success, error, options);
            } else if (error) {
                error({ code: 2, message: 'Position unavailable' });
            }
        }
    };

    // Replace navigator.geolocation
    Object.defineProperty(navigator, 'geolocation', {
        get: function() { return mock; },
        configurable: false
    });

    // Expose for Swift access
    window.__skiframesGeoMock = mock;

    // Override accuracy label to show fix type when available
    document.addEventListener('DOMContentLoaded', function() {
        if (typeof gpsGetAccuracyLabel === 'function') {
            var origLabel = gpsGetAccuracyLabel;
            window.gpsGetAccuracyLabel = function(meters) {
                var pos = window.__skiframesGeoMock._lastPosition;
                if (pos && pos.coords.fixType && pos.coords.fixType >= 2) {
                    var fixLabels = {
                        0: 'No Fix', 1: 'GPS', 2: 'DGPS',
                        4: 'RTK Fixed', 5: 'RTK Float'
                    };
                    var label = fixLabels[pos.coords.fixType] || 'GPS';
                    return '\\u00b1' + meters.toFixed(2) + 'm (' + label + ')';
                }
                return origLabel(meters);
            };
        }
    });
})();
"""

// MARK: - GeolocationWebView (UIViewRepresentable)

struct GeolocationWebView: UIViewRepresentable {
    @ObservedObject var receiver: BadElfReceiver

    func makeCoordinator() -> Coordinator {
        Coordinator(receiver: receiver)
    }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()

        // Inject geolocation mock BEFORE any page script runs
        let mockScript = WKUserScript(
            source: geolocationMockJS,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        config.userContentController.addUserScript(mockScript)

        // Allow inline media playback (for any video in the UI)
        config.allowsInlineMediaPlayback = true

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.scrollView.bounces = false
        webView.allowsBackForwardNavigationGestures = false

        // Use Safari user agent so Google OAuth doesn't block us
        webView.customUserAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1"

        // Enable Web Inspector in debug builds
        #if DEBUG
        if #available(iOS 16.4, *) {
            webView.isInspectable = true
        }
        #endif

        context.coordinator.webView = webView

        // Load the calibration UI
        let url = URL(string: "https://setup.skiframes.com")!
        webView.load(URLRequest(url: url))

        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // updateUIView is called on every SwiftUI state change.
        // The Coordinator handles pushing GPS data via Combine subscriber.
    }

    // MARK: - Coordinator

    class Coordinator: NSObject {
        weak var webView: WKWebView?
        private var cancellable: AnyCancellable?

        init(receiver: BadElfReceiver) {
            super.init()

            // Subscribe to GPS data updates
            cancellable = receiver.$latestData
                .compactMap { $0 }
                .receive(on: DispatchQueue.main)
                .sink { [weak self] data in
                    self?.pushGPSData(data)
                }
        }

        func pushGPSData(_ data: GnssData) {
            let js = """
            if (window.__skiframesGeoMock) {
                window.__skiframesGeoMock._pushPosition({
                    latitude: \(data.latitude),
                    longitude: \(data.longitude),
                    altitude: \(data.altitude),
                    accuracy: \(data.horizontalAccuracy),
                    fixType: \(data.fixType),
                    timestamp: \(data.timestamp.timeIntervalSince1970 * 1000)
                });
            }
            """
            webView?.evaluateJavaScript(js) { _, error in
                if let error = error {
                    print("[Bridge] JS error: \(error.localizedDescription)")
                }
            }
        }
    }
}
