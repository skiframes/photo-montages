/**
 * Pose Analysis Viewer Component
 *
 * Adds "Analyze" buttons to run thumbnails in the gallery and displays
 * body position metrics from MediaPipe pose estimation.
 *
 * Dependencies:
 * - Chart.js (loaded via CDN)
 *
 * Usage:
 *   // Initialize on page load
 *   PoseViewer.init({ edgeApiUrl: 'http://localhost:5000' });
 *
 *   // Add analyze button to a run thumbnail
 *   PoseViewer.addAnalyzeButton(thumbnailElement, {
 *     videoPath: '/path/to/video.mp4',
 *     runId: 'run_001',
 *     cameraId: 'R1'
 *   });
 */

const PoseViewer = (function() {
    'use strict';

    // Configuration
    let config = {
        edgeApiUrl: '',  // Edge device API URL (set via init)
        pollInterval: 1000,  // ms between status polls
    };

    // Active analysis jobs being tracked
    const activeJobs = new Map();

    /**
     * Initialize the pose viewer.
     * @param {Object} options - Configuration options
     * @param {string} options.edgeApiUrl - URL of the edge device API
     */
    function init(options = {}) {
        if (options.edgeApiUrl) {
            config.edgeApiUrl = options.edgeApiUrl.replace(/\/$/, '');
        }

        // Load Chart.js if not already loaded
        if (typeof Chart === 'undefined') {
            loadChartJS();
        }

        console.log('[PoseViewer] Initialized with config:', config);
    }

    /**
     * Load Chart.js from CDN
     */
    function loadChartJS() {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js';
        script.async = true;
        document.head.appendChild(script);
    }

    /**
     * Add an "Analyze" button to a run thumbnail element.
     * @param {HTMLElement} container - The thumbnail container element
     * @param {Object} runInfo - Information about the run
     * @param {string} runInfo.videoPath - Path to the video file
     * @param {string} runInfo.runId - Unique identifier for the run
     * @param {string} [runInfo.cameraId] - Camera ID for slope lookup
     */
    function addAnalyzeButton(container, runInfo) {
        // Check if button already exists
        if (container.querySelector('.pose-analyze-btn')) {
            return;
        }

        const btn = document.createElement('button');
        btn.className = 'pose-analyze-btn';
        btn.innerHTML = '<span class="pose-icon">&#9670;</span> Analyze';
        btn.title = 'Analyze body position';

        // Style the button
        Object.assign(btn.style, {
            position: 'absolute',
            top: '4px',
            left: '4px',
            padding: '4px 8px',
            fontSize: '11px',
            fontWeight: '600',
            background: 'rgba(249, 115, 22, 0.9)',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            zIndex: '10',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
        });

        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            startAnalysis(container, runInfo);
        });

        // Ensure container has relative positioning
        if (getComputedStyle(container).position === 'static') {
            container.style.position = 'relative';
        }

        container.appendChild(btn);
    }

    /**
     * Start pose analysis for a run.
     */
    async function startAnalysis(container, runInfo) {
        const btn = container.querySelector('.pose-analyze-btn');
        if (!btn) return;

        // Update button to show loading
        btn.innerHTML = '<span class="pose-spinner"></span> Analyzing...';
        btn.disabled = true;

        // Add spinner styles if not present
        addSpinnerStyles();

        try {
            const response = await fetch(`${config.edgeApiUrl}/api/analyze/pose`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_path: runInfo.videoPath,
                    run_id: runInfo.runId,
                    camera_id: runInfo.cameraId,
                    sample_rate: 3
                })
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Start polling for status
            pollJobStatus(data.job_id, container, runInfo, btn);

        } catch (err) {
            console.error('[PoseViewer] Analysis failed:', err);
            btn.innerHTML = '<span class="pose-icon">&#9888;</span> Error';
            btn.style.background = '#dc2626';
            btn.disabled = false;

            setTimeout(() => {
                btn.innerHTML = '<span class="pose-icon">&#9670;</span> Retry';
                btn.style.background = 'rgba(249, 115, 22, 0.9)';
            }, 3000);
        }
    }

    /**
     * Poll for analysis job status.
     */
    function pollJobStatus(jobId, container, runInfo, btn) {
        const poll = async () => {
            try {
                const response = await fetch(`${config.edgeApiUrl}/api/analyze/status/${jobId}`);
                const data = await response.json();

                if (data.status === 'running') {
                    // Update progress
                    const progress = data.progress || 0;
                    btn.innerHTML = `<span class="pose-spinner"></span> ${progress.toFixed(0)}%`;

                    // Continue polling
                    activeJobs.set(jobId, setTimeout(poll, config.pollInterval));

                } else if (data.status === 'completed') {
                    // Analysis complete
                    btn.innerHTML = '<span class="pose-icon">&#10003;</span> View Results';
                    btn.style.background = '#16a34a';
                    btn.disabled = false;

                    // Replace click handler to show results
                    btn.onclick = (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        showResults(container, data.summary, runInfo.runId);
                    };

                    activeJobs.delete(jobId);

                    // Auto-show results
                    showResults(container, data.summary, runInfo.runId);

                } else if (data.status === 'error') {
                    throw new Error(data.error || 'Analysis failed');
                }

            } catch (err) {
                console.error('[PoseViewer] Poll error:', err);
                btn.innerHTML = '<span class="pose-icon">&#9888;</span> Error';
                btn.style.background = '#dc2626';
                btn.disabled = false;
                activeJobs.delete(jobId);
            }
        };

        poll();
    }

    /**
     * Show analysis results in a panel.
     */
    function showResults(container, summary, runId) {
        // Remove any existing results panel
        const existing = document.querySelector('.pose-results-panel');
        if (existing) {
            existing.remove();
        }

        // Create results panel
        const panel = document.createElement('div');
        panel.className = 'pose-results-panel';
        Object.assign(panel.style, {
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: 'white',
            borderRadius: '12px',
            boxShadow: '0 4px 24px rgba(0,0,0,0.2)',
            padding: '20px',
            maxWidth: '500px',
            width: '90%',
            maxHeight: '80vh',
            overflow: 'auto',
            zIndex: '10001'
        });

        panel.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h3 style="margin: 0; font-size: 18px;">Body Position Analysis</h3>
                <button class="pose-close-btn" style="background: none; border: none; font-size: 24px; cursor: pointer; color: #64748b;">&times;</button>
            </div>
            <div style="font-size: 12px; color: #64748b; margin-bottom: 16px;">Run: ${runId}</div>

            <div class="pose-metrics-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                ${createMetricCard('Shoulder Angle', summary.avg_shoulder_angle, '°', 'Angle of shoulders relative to slope')}
                ${createMetricCard('Hip Angle', summary.avg_hip_angle, '°', 'Angle of hips relative to slope')}
                ${createMetricCard('Angulation', summary.avg_angulation, '°', 'Knee-hip-shoulder angle (body bend)')}
                ${createMetricCard('Inclination', summary.avg_inclination, '°', 'Overall body lean toward slope')}
            </div>

            <div class="pose-stats" style="background: #f8fafc; border-radius: 8px; padding: 12px; font-size: 13px;">
                <div style="margin-bottom: 8px; font-weight: 600; color: #334155;">Detection Statistics</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; color: #64748b;">
                    <div>Frames analyzed: <span style="color: #1e293b; font-weight: 500;">${summary.frames_with_pose || 0}</span></div>
                    <div>Detection rate: <span style="color: #1e293b; font-weight: 500;">${((summary.detection_rate || 0) * 100).toFixed(0)}%</span></div>
                </div>
            </div>

            <div class="pose-ranges" style="margin-top: 16px;">
                <div style="font-weight: 600; color: #334155; margin-bottom: 8px;">Angle Ranges</div>
                ${createRangeBar('Shoulder', summary.min_shoulder_angle, summary.max_shoulder_angle, summary.avg_shoulder_angle)}
                ${createRangeBar('Hip', summary.min_hip_angle, summary.max_hip_angle, summary.avg_hip_angle)}
                ${createRangeBar('Angulation', summary.min_angulation, summary.max_angulation, summary.avg_angulation)}
                ${createRangeBar('Inclination', summary.min_inclination, summary.max_inclination, summary.avg_inclination)}
            </div>

            <div style="margin-top: 20px;">
                <canvas id="pose-chart" height="200"></canvas>
            </div>
        `;

        // Add backdrop
        const backdrop = document.createElement('div');
        backdrop.className = 'pose-backdrop';
        Object.assign(backdrop.style, {
            position: 'fixed',
            top: '0',
            left: '0',
            right: '0',
            bottom: '0',
            background: 'rgba(0,0,0,0.5)',
            zIndex: '10000'
        });

        document.body.appendChild(backdrop);
        document.body.appendChild(panel);

        // Close button handler
        panel.querySelector('.pose-close-btn').onclick = () => {
            backdrop.remove();
            panel.remove();
        };

        backdrop.onclick = () => {
            backdrop.remove();
            panel.remove();
        };

        // If we have full results, try to fetch and show chart
        fetchAndShowChart(runId);
    }

    /**
     * Create a metric card HTML string.
     */
    function createMetricCard(label, value, unit, tooltip) {
        const displayValue = value !== null && value !== undefined ? value.toFixed(1) : '—';
        return `
            <div class="pose-metric-card" style="background: #f0f9ff; border-radius: 8px; padding: 12px; text-align: center;" title="${tooltip}">
                <div style="font-size: 24px; font-weight: 700; color: #0369a1;">${displayValue}${unit}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">${label}</div>
            </div>
        `;
    }

    /**
     * Create a range bar showing min/max/avg.
     */
    function createRangeBar(label, min, max, avg) {
        if (min === null || max === null) {
            return '';
        }

        // Normalize to a reasonable range for display (-90 to 90 for angles)
        const rangeMin = -90;
        const rangeMax = 90;
        const normalizePos = (val) => ((val - rangeMin) / (rangeMax - rangeMin)) * 100;

        const minPos = Math.max(0, Math.min(100, normalizePos(min)));
        const maxPos = Math.max(0, Math.min(100, normalizePos(max)));
        const avgPos = Math.max(0, Math.min(100, normalizePos(avg)));
        const width = maxPos - minPos;

        return `
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-bottom: 2px;">
                    <span>${label}</span>
                    <span>${min.toFixed(1)}° to ${max.toFixed(1)}°</span>
                </div>
                <div style="position: relative; height: 8px; background: #e2e8f0; border-radius: 4px;">
                    <div style="position: absolute; left: ${minPos}%; width: ${width}%; height: 100%; background: #0ea5e9; border-radius: 4px;"></div>
                    <div style="position: absolute; left: ${avgPos}%; width: 2px; height: 100%; background: #f97316; transform: translateX(-50%);"></div>
                </div>
            </div>
        `;
    }

    /**
     * Fetch full results and show time-series chart.
     */
    async function fetchAndShowChart(runId) {
        try {
            const response = await fetch(`${config.edgeApiUrl}/api/analyze/results/${runId}`);
            if (!response.ok) return;

            const data = await response.json();
            if (!data.frames || data.frames.length === 0) return;

            // Wait for Chart.js to load
            if (typeof Chart === 'undefined') {
                setTimeout(() => fetchAndShowChart(runId), 500);
                return;
            }

            const canvas = document.getElementById('pose-chart');
            if (!canvas) return;

            // Extract time series data
            const timestamps = [];
            const shoulderAngles = [];
            const hipAngles = [];
            const angulations = [];
            const inclinations = [];

            data.frames.forEach(frame => {
                if (frame.pose_detected && frame.metrics) {
                    timestamps.push(frame.timestamp_sec);
                    shoulderAngles.push(frame.metrics.shoulder_angle_to_slope);
                    hipAngles.push(frame.metrics.hip_angle_to_slope);
                    angulations.push(frame.metrics.body_angulation);
                    inclinations.push(frame.metrics.body_inclination);
                }
            });

            new Chart(canvas, {
                type: 'line',
                data: {
                    labels: timestamps.map(t => t.toFixed(2) + 's'),
                    datasets: [
                        {
                            label: 'Shoulder',
                            data: shoulderAngles,
                            borderColor: '#0ea5e9',
                            backgroundColor: 'rgba(14, 165, 233, 0.1)',
                            tension: 0.3,
                            pointRadius: 0
                        },
                        {
                            label: 'Hip',
                            data: hipAngles,
                            borderColor: '#8b5cf6',
                            backgroundColor: 'rgba(139, 92, 246, 0.1)',
                            tension: 0.3,
                            pointRadius: 0
                        },
                        {
                            label: 'Inclination',
                            data: inclinations,
                            borderColor: '#f97316',
                            backgroundColor: 'rgba(249, 115, 22, 0.1)',
                            tension: 0.3,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { boxWidth: 12, font: { size: 11 } }
                        },
                        title: {
                            display: true,
                            text: 'Angles Over Time',
                            font: { size: 13 }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: 'Time (s)', font: { size: 11 } },
                            ticks: { maxTicksLimit: 8, font: { size: 10 } }
                        },
                        y: {
                            display: true,
                            title: { display: true, text: 'Angle (°)', font: { size: 11 } },
                            ticks: { font: { size: 10 } }
                        }
                    }
                }
            });

        } catch (err) {
            console.error('[PoseViewer] Failed to load chart data:', err);
        }
    }

    /**
     * Add CSS styles for the spinner animation.
     */
    function addSpinnerStyles() {
        if (document.querySelector('#pose-viewer-styles')) return;

        const style = document.createElement('style');
        style.id = 'pose-viewer-styles';
        style.textContent = `
            .pose-spinner {
                display: inline-block;
                width: 12px;
                height: 12px;
                border: 2px solid white;
                border-top-color: transparent;
                border-radius: 50%;
                animation: pose-spin 0.8s linear infinite;
            }
            @keyframes pose-spin {
                to { transform: rotate(360deg); }
            }
            .pose-icon {
                font-size: 10px;
            }
        `;
        document.head.appendChild(style);
    }

    // Public API
    return {
        init,
        addAnalyzeButton,
        showResults
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PoseViewer;
}
