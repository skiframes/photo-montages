/**
 * Race Gallery – NHARA Western Division Qualifier GS, Feb 22, 2026
 * 3D terrain + 2D course + results table
 */

// ── State ────────────────────────────────────────────────────────────────
let manifest = null;
let terrainMeta = null;
let leafletMap = null;
let activeView = 'results';
let activeCategory = 'U12_Girls';
let activeRun = 'run1';
let filterTeam = '';
let sortColumn = '';     // '', 'rank', 'time', or 'sectionTime_Cam1', etc.
let sortDirection = '';  // '', 'asc', 'desc'

// Three.js (loaded via import map)
let THREE, OrbitControls;
let scene, camera3d, renderer, controls;
let worldGroup; // group containing terrain + all overlays (scaled for flatten)
let terrainMesh, gateMeshes = [], camMeshes = [], sectionMeshes = [];
let labelSprites = [];
let trailBoundaryMeshes = [];
let showCameras = false;
let camFovMeshes = []; // 3D FOV cone meshes (toggled with camera checkbox)
let terrainTexture = null;   // 1m hillshade texture
let satelliteTexture = null; // Esri satellite imagery texture
let terrainStyle = 'satellite'; // 'satellite' or 'terrain'
let isFlattened = false; // toggle between 3D terrain and flat 2D view
let showMeasurements = false; // toggle 2D measurement labels (unchecked by default)
let terrainInfoSprites = []; // elevation + gate measurement labels

const VERT_SCALE = 1.0;

// Section colors (one per camera)
const SECTION_COLORS = { 'Cam1': 0xf59e0b, 'Cam2': 0x0891b2, 'Cam3': 0x8b5cf6 };
const SECTION_COLORS_CSS = { 'Cam1': '#f59e0b', 'Cam2': '#0891b2', 'Cam3': '#8b5cf6' };

// Raycaster for gate hover tooltips
let raycaster, mouse;
let gateHoverTargets = [];
let sectionLabelTargets = []; // clickable section labels in 3D view

/**
 * Get per-run camera coverage for the active run.
 * Falls back to gates_covered if per-run fields don't exist.
 */
function getCamCoverage(cam) {
    const runKey = 'gates_covered_' + activeRun;  // gates_covered_run1 or gates_covered_run2
    return cam[runKey] || cam.gates_covered || [];
}

// ── Init ─────────────────────────────────────────────────────────────────
async function init() {
    console.log('[race] init starting...');

    try {
        const [mResp, tResp] = await Promise.all([
            fetch('race_manifest.json?v=' + Date.now()),
            fetch('terrain_meta.json?v2'),
        ]);
        manifest = await mResp.json();
        terrainMeta = await tResp.json();

        // When served locally (localhost, 127.0.0.1, or file://), use local montage server
        const host = window.location.hostname;
        const isLocal = host === 'localhost' || host === '127.0.0.1' || window.location.protocol === 'file:';
        if (isLocal) {
            const raceSlug = manifest.media_base_url.split('/').pop();
            manifest.media_base_url = `http://localhost:5000/montages/${raceSlug}`;
            console.log('[race] Local mode — media_base_url overridden to:', manifest.media_base_url);
        }

        // Normalize montages: convert old single-object format to array format
        // Old: montages.Cam1.run1 = {thumb, full, ...}
        // New: montages.Cam1.run1 = [{det_id, thumb, full, ...}, ...]
        for (const cat of (manifest.categories || [])) {
            for (const ath of (cat.athletes || [])) {
                if (!ath.montages) continue;
                for (const camId of Object.keys(ath.montages)) {
                    for (const runKey of Object.keys(ath.montages[camId])) {
                        const val = ath.montages[camId][runKey];
                        if (val && !Array.isArray(val)) {
                            // Legacy single object — wrap in array
                            if (!val.det_id) val.det_id = 'd000';
                            ath.montages[camId][runKey] = [val];
                        }
                    }
                }
            }
        }

        console.log('[race] manifest loaded:', manifest.categories.length, 'categories');
        console.log('[race] terrain:', terrainMeta.width + 'x' + terrainMeta.height, 'at', terrainMeta.resolution_m + 'm resolution');
    } catch (e) {
        console.error('[race] Failed to load data:', e);
        document.body.innerHTML = '<div style="padding:40px;color:#ef4444;">Error loading data: ' + e.message + '</div>';
        return;
    }

    setupViewTabs();
    setupCategoryTabs();
    setupLightbox();
    setupVideoLightbox();
    setupTeamFilter();
    setupSearch();
    setupDocsScrollSpy();
    renderResults();
    console.log('[race] UI setup complete');

    init3D().catch(e => {
        console.error('[race] 3D init failed:', e);
        document.getElementById('three-container').innerHTML =
            '<div style="padding:40px;color:#ef4444;">3D view error: ' + e.message + '</div>';
    });
}

// ── View Switching ───────────────────────────────────────────────────────
function setupViewTabs() {
    document.querySelectorAll('#view-tabs .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const view = tab.dataset.view;
            document.querySelectorAll('#view-tabs .tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById(view + '-view').classList.add('active');
            activeView = view;
            if (view === 'map3d') {
                if (renderer) {
                    onResize3D();
                } else {
                    // Deferred init: 3D container was hidden at page load so init was skipped
                    init3D().catch(e => {
                        console.error('[race] 3D init failed:', e);
                        document.getElementById('three-container').innerHTML =
                            '<div style="padding:40px;color:#ef4444;">3D view error: ' + e.message + '</div>';
                    });
                }
            }
            if (view === 'map2d') {
                if (!leafletMap) initLeaflet();
                else setTimeout(() => leafletMap.invalidateSize(), 100);
                // Focus the map container so Leaflet keyboard controls work
                setTimeout(() => {
                    const mapEl = document.getElementById('map');
                    if (mapEl) mapEl.focus();
                }, 200);
            }
        });
    });
}

function setupCategoryTabs() {
    document.querySelectorAll('.cat-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeCategory = tab.dataset.cat;
            sortColumn = ''; sortDirection = '';
            renderResults();
        });
    });

    // Run sub-tabs in results view
    document.querySelectorAll('.run-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.run-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeRun = tab.dataset.run;
            // Sync with map run selectors
            document.getElementById('run-select-3d').value = activeRun;
            document.getElementById('run-select-2d').value = activeRun;
            // Update maps
            if (scene) updateCourseOverlay();
            if (leafletMap) updateLeafletOverlay();
            renderResults();
        });
    });
}

/**
 * Sync all run selectors/tabs to current activeRun value.
 */
function syncRunTabs() {
    // Sync map dropdowns
    const sel3d = document.getElementById('run-select-3d');
    const sel2d = document.getElementById('run-select-2d');
    if (sel3d) sel3d.value = activeRun;
    if (sel2d) sel2d.value = activeRun;
    // Sync results run tabs
    document.querySelectorAll('.run-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.run === activeRun);
    });
    // Re-render results for new run
    renderResults();
}

function setupTeamFilter() {
    // Collect all unique team names across all categories
    const teams = new Set();
    manifest.categories.forEach(cat => {
        cat.athletes.forEach(a => { if (a.club) teams.add(a.club); });
    });
    const sorted = [...teams].sort();
    const sel = document.getElementById('team-select');
    sorted.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        sel.appendChild(opt);
    });
    sel.addEventListener('change', (e) => {
        filterTeam = e.target.value;
        renderResults();
    });

    // Team download dropdown - triggers download directly on selection
    const dlSel = document.getElementById('team-download-select');
    if (dlSel) {
        sorted.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            dlSel.appendChild(opt);
        });
        dlSel.addEventListener('change', () => {
            const team = dlSel.value;
            if (!team) return;
            downloadTeamMedia(team);
            dlSel.value = '';  // Reset dropdown after selection
        });
    }
}

async function downloadTeamMedia(team) {
    // Show disclaimer modal
    const proceed = confirm(
        `Download all media for ${team}?\n\n` +
        `⚠️ WARNING: This download may be several GB in size.\n\n` +
        `• Use a desktop computer, not a mobile phone\n` +
        `• Ensure you have a stable internet connection\n` +
        `• Make sure you have enough disk space\n\n` +
        `The download will include all photo montages and video clips for athletes from ${team}.`
    );
    if (!proceed) return;

    // Collect all media URLs for the team
    const mediaUrls = [];
    manifest.categories.forEach(cat => {
        cat.athletes.forEach(a => {
            if (a.club !== team || !a.montages) return;
            for (const camId of Object.keys(a.montages)) {
                for (const runKey of Object.keys(a.montages[camId])) {
                    const dets = a.montages[camId][runKey];
                    if (!Array.isArray(dets)) continue;
                    dets.forEach(det => {
                        // Add full-res image
                        if (det.full) {
                            mediaUrls.push({
                                url: manifest.media_base_url + '/' + det.full,
                                filename: `${a.last}_${a.first}_${a.bib}/${camId}_${runKey}_${det.full.split('/').pop()}`
                            });
                        }
                        // Add video
                        if (det.video) {
                            mediaUrls.push({
                                url: manifest.media_base_url + '/' + det.video,
                                filename: `${a.last}_${a.first}_${a.bib}/${camId}_${runKey}_${det.video.split('/').pop()}`
                            });
                        }
                    });
                }
            }
        });
    });

    if (mediaUrls.length === 0) {
        alert(`No media found for ${team}.`);
        return;
    }

    // For now, open each file in new tab (browsers will prompt to download)
    // A proper implementation would use a server-side zip endpoint
    alert(
        `Found ${mediaUrls.length} files for ${team}.\n\n` +
        `To download all files, please use the browser's download manager or ` +
        `contact the race organizer for a bulk download link.`
    );

    // Open the team-filtered results page so they can download individually
    document.getElementById('team-select').value = team;
    filterTeam = team;
    renderResults();
}


// ══════════════════════════════════════════════════════════════════════════
// SEARCH
// ══════════════════════════════════════════════════════════════════════════

function setupSearch() {
    const input = document.getElementById('search-input');
    const resultsDiv = document.getElementById('search-results');

    input.addEventListener('input', () => {
        const q = input.value.trim().toLowerCase();
        if (q.length < 1) {
            resultsDiv.classList.add('hidden');
            resultsDiv.innerHTML = '';
            return;
        }

        // Search across all categories
        const matches = [];
        manifest.categories.forEach(cat => {
            cat.athletes.forEach(a => {
                const fullName = (a.first + ' ' + a.last).toLowerCase();
                const bibStr = String(a.bib);
                if (fullName.includes(q) || bibStr.startsWith(q)) {
                    matches.push({ athlete: a, cat });
                }
            });
        });

        if (matches.length === 0) {
            resultsDiv.innerHTML = '<div class="search-item"><span class="search-meta">No results found</span></div>';
            resultsDiv.classList.remove('hidden');
            return;
        }

        // Limit to 20 results
        const shown = matches.slice(0, 20);
        resultsDiv.innerHTML = shown.map(({ athlete: a, cat }) =>
            `<div class="search-item" data-bib="${a.bib}" data-cat="${cat.id}">
                <span class="search-bib">#${a.bib}</span>
                <span class="search-name">${a.first} ${a.last}</span>
                <span class="search-meta">${a.club} &middot; ${cat.label}</span>
            </div>`
        ).join('');
        resultsDiv.classList.remove('hidden');
    });

    // Click on search result → switch to results view, select category, highlight
    resultsDiv.addEventListener('click', (e) => {
        const item = e.target.closest('.search-item');
        if (!item || !item.dataset.bib) return;

        const catId = item.dataset.cat;
        const bib = parseInt(item.dataset.bib);

        // Switch to results view
        document.querySelectorAll('#view-tabs .tab').forEach(t => t.classList.remove('active'));
        document.querySelector('#view-tabs .tab[data-view="results"]').classList.add('active');
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById('results-view').classList.add('active');
        activeView = 'results';

        // Switch category
        activeCategory = catId;
        document.querySelectorAll('.cat-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.cat === catId);
        });

        // Clear team filter
        filterTeam = '';
        document.getElementById('team-select').value = '';

        renderResults();

        // Scroll to the athlete row
        setTimeout(() => {
            const row = document.querySelector(`#results-body tr[data-bib="${bib}"]`);
            if (row) {
                row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                row.classList.add('highlight');
                setTimeout(() => row.classList.remove('highlight'), 2000);
            }
        }, 100);

        // Close search
        resultsDiv.classList.add('hidden');
        input.value = '';
    });

    // Close on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('#search-bar')) {
            resultsDiv.classList.add('hidden');
        }
    });

    // Close on Escape
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            resultsDiv.classList.add('hidden');
            input.blur();
        }
    });
}


// ══════════════════════════════════════════════════════════════════════════
// DOCUMENTATION SCROLL-SPY
// ══════════════════════════════════════════════════════════════════════════

function setupDocsScrollSpy() {
    const links = document.querySelectorAll('.docs-nav-link');
    const articles = document.querySelectorAll('.docs-article');
    const content = document.querySelector('.docs-content');
    if (!content || articles.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                links.forEach(l => l.classList.remove('active'));
                const link = document.querySelector(`.docs-nav-link[href="#${entry.target.id}"]`);
                if (link) link.classList.add('active');
            }
        });
    }, { root: content, rootMargin: '-10% 0px -80% 0px', threshold: 0 });

    articles.forEach(a => observer.observe(a));

    // Sidebar click: smooth scroll to article
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.getElementById(link.getAttribute('href').slice(1));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}


// ══════════════════════════════════════════════════════════════════════════
// THREE.JS 3D TERRAIN
// ══════════════════════════════════════════════════════════════════════════

async function init3D() {
    console.log('[race] loading Three.js...');
    THREE = await import('three');
    const OrbitMod = await import('three/addons/controls/OrbitControls.js');
    OrbitControls = OrbitMod.OrbitControls;
    console.log('[race] Three.js loaded');

    const container = document.getElementById('three-container');
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xd4e6f1);
    scene.fog = new THREE.Fog(0xd4e6f1, 800, 1800);

    camera3d = new THREE.PerspectiveCamera(50, w / h, 0.1, 3000);
    camera3d.position.set(60, 54, -266);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera3d, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxPolarAngle = Math.PI / 2.2;
    controls.minDistance = 1;
    controls.maxDistance = 1200;
    controls.zoomSpeed = 2.0;     // faster scroll zoom (default 1.0)
    controls.panSpeed = 1.5;      // faster right-drag pan
    controls.rotateSpeed = 0.8;
    controls.target.set(99, 26, -73);
    controls.listenToKeyEvents(window); // enable arrow keys + +/- for pan/zoom
    controls.keys = { LEFT: 'ArrowLeft', UP: 'ArrowUp', RIGHT: 'ArrowRight', BOTTOM: 'ArrowDown' };
    controls.keyPanSpeed = 15;    // faster arrow-key panning (default 7)

    // World group: contains terrain + all course overlays. Scaled for flatten.
    worldGroup = new THREE.Group();
    scene.add(worldGroup);

    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xfffff0, 0.85);
    sun.position.set(-150, 300, 150);
    sun.castShadow = true;
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0xddeeff, 0.3);
    fill.position.set(100, 100, -100);
    scene.add(fill);

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    await loadTerrain();
    addTrailBoundary();
    updateCourseOverlay();

    document.getElementById('run-select-3d').addEventListener('change', (e) => {
        activeRun = e.target.value;
        updateCourseOverlay();
        syncRunTabs();
    });

    // Terrain style toggle (satellite / 1m terrain)
    document.getElementById('terrain-style-3d').addEventListener('change', (e) => {
        setTerrainStyle(e.target.value);
    });

    // Flatten toggle button
    document.getElementById('flatten-btn').addEventListener('click', () => {
        isFlattened = !isFlattened;
        document.getElementById('flatten-btn').classList.toggle('active', isFlattened);
        document.getElementById('flatten-btn').textContent = isFlattened ? '▦ 3D' : '▦ Flatten';
        animateFlatten(isFlattened);
    });

    // Camera toggle checkbox (3D)
    document.getElementById('show-cameras-3d').addEventListener('change', (e) => {
        showCameras = e.target.checked;
        // Sync 2D checkbox
        const cb2d = document.getElementById('show-cameras-2d');
        if (cb2d) cb2d.checked = showCameras;
        toggleCameras3D();
        if (leafletMap) toggleCameras2D();
    });

    container.addEventListener('mousemove', onMouseMove3D);
    container.addEventListener('mouseleave', () => {
        document.getElementById('info-tooltip').classList.add('hidden');
        container.style.cursor = '';
    });
    container.addEventListener('click', onClick3D);

    window.addEventListener('resize', onResize3D);
    // Debug: press 'C' to log camera position for setting defaults
    window.addEventListener('keydown', (e) => {
        if (e.key === 'c' || e.key === 'C') {
            const p = camera3d.position;
            const t = controls.target;
            console.log(`[camera] position.set(${p.x.toFixed(0)}, ${p.y.toFixed(0)}, ${p.z.toFixed(0)})  target.set(${t.x.toFixed(0)}, ${t.y.toFixed(0)}, ${t.z.toFixed(0)})`);
        }
    });
    animate();
    console.log('[race] 3D init complete');
}

function onResize3D() {
    const container = document.getElementById('three-container');
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    camera3d.aspect = w / h;
    camera3d.updateProjectionMatrix();
    renderer.setSize(w, h);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera3d);
}

function onMouseMove3D(event) {
    const container = document.getElementById('three-container');
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera3d);
    const meshes = gateHoverTargets.map(t => t.mesh);
    const intersects = raycaster.intersectObjects(meshes, false);

    const tooltip = document.getElementById('info-tooltip');
    if (intersects.length > 0) {
        const hit = intersects[0].object;
        const target = gateHoverTargets.find(t => t.mesh === hit);
        if (target) {
            tooltip.innerHTML = target.label;
            tooltip.style.left = (event.clientX - rect.left + 12) + 'px';
            tooltip.style.top = (event.clientY - rect.top - 8) + 'px';
            tooltip.classList.remove('hidden');
        }
    } else {
        tooltip.classList.add('hidden');
    }

    // Check section label hover for pointer cursor
    if (sectionLabelTargets.length > 0) {
        const sectionSprites = sectionLabelTargets.map(t => t.sprite);
        const sectionHits = raycaster.intersectObjects(sectionSprites, false);
        container.style.cursor = sectionHits.length > 0 ? 'pointer' : '';
    }
}

function onClick3D(event) {
    if (!raycaster || !camera3d || sectionLabelTargets.length === 0) return;
    const container = document.getElementById('three-container');
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera3d);
    const sprites = sectionLabelTargets.map(t => t.sprite);
    const intersects = raycaster.intersectObjects(sprites, false);

    if (intersects.length > 0) {
        const hit = intersects[0].object;
        const target = sectionLabelTargets.find(t => t.sprite === hit);
        if (target) {
            showSectionPopup(target.camId);
        }
    }
}

async function loadTerrain() {
    const tm = terrainMeta;

    const cacheBust = 'v3';  // bump to force reload after terrain regeneration
    const hmResp = await fetch('terrain_heightmap.bin?' + cacheBust);
    const hmBuf = await hmResp.arrayBuffer();
    const heightData = new Uint16Array(hmBuf);

    // Load 1m hillshade texture
    const texLoader = new THREE.TextureLoader();
    terrainTexture = await new Promise((resolve, reject) => {
        texLoader.load('terrain_texture.png?' + cacheBust, resolve, undefined, reject);
    });
    terrainTexture.minFilter = THREE.LinearFilter;
    terrainTexture.magFilter = THREE.LinearFilter;

    // Build geometry
    const geo = new THREE.PlaneGeometry(
        tm.extent_m * 2, tm.extent_m * 2,
        tm.width - 1, tm.height - 1
    );

    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
        const elev = heightData[i] * tm.elev_scale + tm.elev_min;
        pos.setZ(i, (elev - tm.elev_min) * VERT_SCALE);
    }
    geo.computeVertexNormals();

    // Start with satellite as default; use terrain as fallback while satellite loads
    const mat = new THREE.MeshLambertMaterial({ map: terrainTexture, side: THREE.DoubleSide });
    terrainMesh = new THREE.Mesh(geo, mat);
    terrainMesh.rotation.x = -Math.PI / 2;
    terrainMesh.receiveShadow = true;
    worldGroup.add(terrainMesh);

    // Load satellite texture in background
    loadSatelliteTexture();
}

/**
 * Fetch Esri satellite tiles and composite into a single texture for the 3D terrain.
 * Uses a canvas to stitch tiles covering the terrain bounding box.
 */
async function loadSatelliteTexture() {
    const tm = terrainMeta;
    const bounds = tm.bounds;

    // Use zoom level 19 for high detail (~0.3m/px at lat 43, Google Earth quality)
    const zoom = 19;

    // Convert lat/lon to tile coordinates
    function latLonToTile(lat, lon, z) {
        const n = Math.pow(2, z);
        const x = Math.floor((lon + 180) / 360 * n);
        const latRad = lat * Math.PI / 180;
        const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
        return { x, y };
    }

    // Convert tile coordinates back to lat/lon (NW corner of tile)
    function tileToLatLon(tx, ty, z) {
        const n = Math.pow(2, z);
        const lon = tx / n * 360 - 180;
        const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / n)));
        const lat = latRad * 180 / Math.PI;
        return { lat, lon };
    }

    const tileNW = latLonToTile(bounds.north, bounds.west, zoom);
    const tileSE = latLonToTile(bounds.south, bounds.east, zoom);

    const tilesX = tileSE.x - tileNW.x + 1;
    const tilesY = tileSE.y - tileNW.y + 1;
    const tileSize = 256;

    // Create canvas to composite all tiles
    const canvas = document.createElement('canvas');
    canvas.width = tilesX * tileSize;
    canvas.height = tilesY * tileSize;
    const ctx = canvas.getContext('2d');

    // Fetch all tiles in parallel
    const tilePromises = [];
    for (let ty = tileNW.y; ty <= tileSE.y; ty++) {
        for (let tx = tileNW.x; tx <= tileSE.x; tx++) {
            const url = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${ty}/${tx}`;
            const px = (tx - tileNW.x) * tileSize;
            const py = (ty - tileNW.y) * tileSize;
            tilePromises.push(
                new Promise((resolve) => {
                    const img = new Image();
                    img.crossOrigin = 'anonymous';
                    img.onload = () => { ctx.drawImage(img, px, py); resolve(); };
                    img.onerror = () => { resolve(); }; // skip failed tiles
                    img.src = url;
                })
            );
        }
    }

    await Promise.all(tilePromises);
    console.log(`[race] satellite: ${tilesX}x${tilesY} tiles at zoom ${zoom}`);

    // Now crop the canvas to exactly match the terrain bounds
    // Compute pixel positions of terrain bounds within the tile grid
    const gridNW = tileToLatLon(tileNW.x, tileNW.y, zoom);
    const gridSE = tileToLatLon(tileSE.x + 1, tileSE.y + 1, zoom);

    const cropLeft = (bounds.west - gridNW.lon) / (gridSE.lon - gridNW.lon) * canvas.width;
    const cropTop = (gridNW.lat - bounds.north) / (gridNW.lat - gridSE.lat) * canvas.height;
    const cropRight = (bounds.east - gridNW.lon) / (gridSE.lon - gridNW.lon) * canvas.width;
    const cropBottom = (gridNW.lat - bounds.south) / (gridNW.lat - gridSE.lat) * canvas.height;

    const cropW = cropRight - cropLeft;
    const cropH = cropBottom - cropTop;

    // Create cropped canvas at high resolution
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = 2048;  // high-res output texture
    croppedCanvas.height = 2048;
    const cctx = croppedCanvas.getContext('2d');
    cctx.drawImage(canvas, cropLeft, cropTop, cropW, cropH, 0, 0, 2048, 2048);

    // Create Three.js texture from cropped canvas
    satelliteTexture = new THREE.CanvasTexture(croppedCanvas);
    satelliteTexture.minFilter = THREE.LinearFilter;
    satelliteTexture.magFilter = THREE.LinearFilter;

    // Apply satellite by default
    if (terrainStyle === 'satellite' && terrainMesh) {
        terrainMesh.material.map = satelliteTexture;
        terrainMesh.material.needsUpdate = true;
    }

    console.log('[race] satellite texture ready');
}

function setTerrainStyle(style) {
    terrainStyle = style;
    if (!terrainMesh) return;
    if (style === 'satellite' && satelliteTexture) {
        terrainMesh.material.map = satelliteTexture;
    } else {
        terrainMesh.material.map = terrainTexture;
    }
    terrainMesh.material.needsUpdate = true;

    // Hide B-net lines on satellite (they clutter the satellite view)
    const showBnets = (style !== 'satellite');
    trailBoundaryMeshes.forEach(m => { m.visible = showBnets; });
}

/**
 * Smoothly animate between 3D terrain and flat 2D-like view.
 * Flattens terrain geometry and rebuilds overlays at flat positions.
 * South-up orientation: start (positive Z) at top of screen, finish (negative Z) at bottom.
 */
let flattenAnim = null;
let flattenFactor = 0; // 0 = full 3D, 1 = fully flat
const savedCamera = { pos: null, target: null }; // save 3D camera state before flattening
let terrainOrigZ = null; // original Z values of terrain vertices (PlaneGeometry Z = elevation)

function animateFlatten(flatten) {
    if (flattenAnim) cancelAnimationFrame(flattenAnim);

    const duration = 600; // ms
    const startTime = performance.now();
    const fromFactor = flattenFactor;
    const toFactor = flatten ? 1 : 0;

    // Save original terrain vertex Z values (elevation) on first flatten
    if (!terrainOrigZ && terrainMesh) {
        const pos = terrainMesh.geometry.attributes.position;
        terrainOrigZ = new Float32Array(pos.count);
        for (let i = 0; i < pos.count; i++) terrainOrigZ[i] = pos.getZ(i);
    }

    // Save camera state before flattening
    if (flatten) {
        savedCamera.pos = camera3d.position.clone();
        savedCamera.target = controls.target.clone();
    }

    // Course center for top-down view
    const courseCX = 4.5, courseCZ = 4.2;

    // Target camera: top-down for flat, restored perspective for 3D
    const targetTarget = flatten
        ? new THREE.Vector3(courseCX, 0, courseCZ)
        : (savedCamera.target || new THREE.Vector3(99, 26, -73));
    const targetPos = flatten
        ? new THREE.Vector3(courseCX, 500, courseCZ - 0.1)
        : (savedCamera.pos || new THREE.Vector3(60, 54, -266));

    const fromPos = camera3d.position.clone();
    const fromTarget = controls.target.clone();

    function step(now) {
        const t = Math.min((now - startTime) / duration, 1);
        // Ease in-out
        const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        flattenFactor = fromFactor + (toFactor - fromFactor) * ease;

        // Interpolate terrain vertex elevations
        if (terrainOrigZ && terrainMesh) {
            const pos = terrainMesh.geometry.attributes.position;
            for (let i = 0; i < pos.count; i++) {
                pos.setZ(i, terrainOrigZ[i] * (1 - flattenFactor));
            }
            pos.needsUpdate = true;
            terrainMesh.geometry.computeVertexNormals();
        }

        camera3d.position.lerpVectors(fromPos, targetPos, ease);
        controls.target.lerpVectors(fromTarget, targetTarget, ease);
        controls.update();

        if (t < 1) {
            flattenAnim = requestAnimationFrame(step);
        } else {
            flattenAnim = null;
            // Rebuild course overlays at final positions
            updateCourseOverlay();
            addTrailBoundary();
            // Lock polar angle when flattened (top-down only)
            if (flatten) {
                controls.maxPolarAngle = 0.01;
                controls.minPolarAngle = 0;
            } else {
                controls.maxPolarAngle = Math.PI / 2.2;
                controls.minPolarAngle = 0;
            }
        }
    }

    // Release polar angle lock before animating back to 3D
    if (!flatten) {
        controls.maxPolarAngle = Math.PI / 2.2;
        controls.minPolarAngle = 0;
    }

    flattenAnim = requestAnimationFrame(step);
}

/**
 * Focus the 3D camera on a specific course section.
 * Switches to the 3D map tab, then animates camera to look at the
 * center of the section's covered gates from a good viewing angle.
 * @param {string} camId - Camera ID, e.g. 'Cam1', 'Cam2', 'Cam3'
 */
let sectionFocusAnim = null;
window.focusOnSection = function focusOnSection(camId) {
    // Switch to 3D map tab
    const map3dTab = document.querySelector('#view-tabs .tab[data-view="map3d"]');
    if (map3dTab) map3dTab.click();

    // Wait for 3D to be ready then animate
    const doFocus = () => {
        if (!camera3d || !controls || !manifest || !manifest.course || !terrainMeta) {
            console.warn('[race] 3D not ready for section focus');
            return;
        }

        // If currently flattened, un-flatten first
        if (isFlattened) {
            isFlattened = false;
            animateFlatten(false);
        }

        const cam = manifest.cameras.find(c => c.id === camId);
        if (!cam) { console.warn('[race] Camera not found:', camId); return; }

        const coverage = getCamCoverage(cam);
        if (coverage.length === 0) return;

        const course = manifest.course[activeRun] || manifest.course.run1;
        const gates = course.gates || [];
        const coveredGates = gates.filter(g => coverage.includes(g.number));
        if (coveredGates.length === 0) return;

        // Compute center of covered gates in 3D
        const positions = coveredGates.map(g => geoTo3D(g.lat, g.lon, g.dem_elev || g.elev));
        let cx = 0, cy = 0, cz = 0;
        positions.forEach(p => { cx += p.x; cy += p.y; cz += p.z; });
        cx /= positions.length;
        cy /= positions.length;
        cz /= positions.length;

        // Target = center of covered gates (no shift)
        const targetPoint = new THREE.Vector3(cx, cy, cz);

        // Compute viewing angle: offset from center, elevated, looking at gates
        // Use camera physical position if available for viewing direction hint
        let viewOffsetX = 40, viewOffsetZ = -60;
        if (cam.position && cam.position.lat) {
            const camPos3d = geoTo3D(cam.position.lat, cam.position.lon,
                cam.position.dem_elev || cam.position.elev || terrainMeta.elev_min + 50);
            // Direction from gate center toward camera, but at a reasonable distance
            const dx = camPos3d.x - cx;
            const dz = camPos3d.z - cz;
            const dist = Math.sqrt(dx * dx + dz * dz) || 1;
            // Normalize and scale to a good viewing distance
            viewOffsetX = (dx / dist) * 80;
            viewOffsetZ = (dz / dist) * 80;
        }

        // Position camera at moderate height — lower = less tilt = more uphill (START) visible
        const viewPos = new THREE.Vector3(cx + viewOffsetX, cy + 25, cz + viewOffsetZ);

        // Animate camera
        if (sectionFocusAnim) cancelAnimationFrame(sectionFocusAnim);
        const duration = 800;
        const startTime = performance.now();
        const fromPos = camera3d.position.clone();
        const fromTarget = controls.target.clone();

        function step(now) {
            const t = Math.min((now - startTime) / duration, 1);
            const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

            camera3d.position.lerpVectors(fromPos, viewPos, ease);
            controls.target.lerpVectors(fromTarget, targetPoint, ease);
            controls.update();

            if (t < 1) {
                sectionFocusAnim = requestAnimationFrame(step);
            } else {
                sectionFocusAnim = null;
            }
        }
        sectionFocusAnim = requestAnimationFrame(step);
    };

    // If 3D not yet initialized, poll until ready (init3D is async)
    if (!renderer || !camera3d) {
        let retries = 0;
        const poll = () => {
            if (renderer && camera3d && controls) {
                doFocus();
            } else if (retries < 20) {
                retries++;
                setTimeout(poll, 200);
            } else {
                console.warn('[race] 3D did not initialize in time for section focus');
            }
        };
        setTimeout(poll, 300);
    } else {
        doFocus();
    }
}

window.focusOnSection2D = function focusOnSection2D(camId) {
    // Switch to 2D map tab
    const map2dTab = document.querySelector('#view-tabs .tab[data-view="map2d"]');
    if (map2dTab) map2dTab.click();

    const doFocus = () => {
        if (!leafletMap || !manifest || !manifest.course) return;

        const cam = manifest.cameras.find(c => c.id === camId);
        if (!cam) return;

        const coverage = getCamCoverage(cam);
        if (coverage.length === 0) return;

        const course = manifest.course[activeRun] || manifest.course.run1;
        const gates = course.gates || [];
        const coveredGates = gates.filter(g => coverage.includes(g.number));
        if (coveredGates.length === 0) return;

        // Build bounds from covered gates + camera position
        const bounds = coveredGates.map(g => [g.lat, g.lon]);
        if (cam.position && cam.position.lat) {
            bounds.push([cam.position.lat, cam.position.lon]);
        }
        leafletMap.flyToBounds(bounds, { padding: [60, 60], maxZoom: 19, duration: 0.8 });
    };

    // Leaflet may not be ready yet if tab was never opened
    if (!leafletMap) {
        setTimeout(doFocus, 500);
    } else {
        // Small delay for tab switch animation
        setTimeout(doFocus, 100);
    }
}

// ── Section Trigger Zone Popup ──────────────────────────────────────────
window.showSectionPopup = function showSectionPopup(camId) {
    const cam = manifest.cameras.find(c => c.id === camId);
    if (!cam) return;
    // Support per-run trigger_zone_images (new) or single trigger_zone_image (legacy)
    const zoneImages = cam.trigger_zone_images || {};
    const zoneImage = zoneImages[activeRun] || zoneImages['run1'] || cam.trigger_zone_image;
    if (!zoneImage) return;

    const idx = manifest.cameras.indexOf(cam) + 1;
    const coverage = getCamCoverage(cam);
    const gatesStr = coverage.length > 0 ? `Gates ${coverage.join(', ')}` : '';
    const runLabel = activeRun.replace('run', 'Run ');

    document.getElementById('sp-title').textContent =
        `Section ${idx} ${runLabel} \u2014 Camera ${cam.edge_camera} (${gatesStr})`;
    document.getElementById('sp-img').src = zoneImage;
    document.getElementById('sp-info').textContent = cam.note || '';
    document.getElementById('section-popup').classList.remove('hidden');
}

function closeSectionPopup() {
    document.getElementById('section-popup').classList.add('hidden');
}

// Bind popup close handlers once DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const popup = document.getElementById('section-popup');
    if (!popup) return;
    popup.querySelector('.sp-backdrop').addEventListener('click', closeSectionPopup);
    popup.querySelector('.sp-close').addEventListener('click', closeSectionPopup);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !popup.classList.contains('hidden')) {
            closeSectionPopup();
        }
    });
});

function getTerrainElevAt(x, z) {
    const tm = terrainMeta;
    const halfExtent = tm.extent_m;
    const col = Math.round(((x + halfExtent) / (2 * halfExtent)) * (tm.width - 1));
    const row = Math.round(((z + halfExtent) / (2 * halfExtent)) * (tm.height - 1));
    if (col < 0 || col >= tm.width || row < 0 || row >= tm.height) return 0;
    if (!terrainMesh) return 0;
    const idx = row * tm.width + col;
    const pos = terrainMesh.geometry.attributes.position;
    if (idx < 0 || idx >= pos.count) return 0;
    return pos.getZ(idx);
}

/**
 * Trail boundary: two solid orange lines along the edges of the trail,
 * offset from the race center line. Represents B-net fencing.
 */
function addTrailBoundary() {
    // Clear existing boundary meshes
    trailBoundaryMeshes.forEach(m => worldGroup.remove(m));
    trailBoundaryMeshes = [];

    const course = manifest.course.run2;
    if (!course || !course.gates || course.gates.length < 2) return;

    const allPts = [];
    if (course.start) allPts.push(course.start);
    course.gates.forEach(g => allPts.push(g));
    // Don't extend to finish — just use gates

    const centerLine = allPts.map(g => geoTo3D(g.lat, g.lon, g.dem_elev || g.elev));

    // Trail half-width widens slightly toward the bottom
    const widthTop = 20, widthBot = 30;

    const leftEdge = [];
    const rightEdge = [];

    for (let i = 0; i < centerLine.length; i++) {
        const pt = centerLine[i];
        let dir;
        if (i < centerLine.length - 1) {
            dir = new THREE.Vector3().subVectors(centerLine[i + 1], pt).normalize();
        } else {
            dir = new THREE.Vector3().subVectors(pt, centerLine[i - 1]).normalize();
        }
        const perp = new THREE.Vector3(-dir.z, 0, dir.x).normalize();
        const t = i / (centerLine.length - 1);
        const halfW = widthTop + (widthBot - widthTop) * t;

        const lp = new THREE.Vector3(pt.x - perp.x * halfW, 0, pt.z - perp.z * halfW);
        const rp = new THREE.Vector3(pt.x + perp.x * halfW, 0, pt.z + perp.z * halfW);
        const ly = getTerrainElevAt(lp.x, lp.z);
        const ry = getTerrainElevAt(rp.x, rp.z);
        lp.y = (ly > 0 ? ly : pt.y) + 0.5;
        rp.y = (ry > 0 ? ry : pt.y) + 0.5;
        leftEdge.push(lp);
        rightEdge.push(rp);
    }

    // Subdivide for smoother draping
    const smoothLeft = subdivideAndDrape(leftEdge);
    const smoothRight = subdivideAndDrape(rightEdge);

    // Draw as solid orange lines (B-net fencing)
    const bnetColor = 0xf97316; // orange
    [smoothLeft, smoothRight].forEach(edge => {
        if (edge.length < 2) return;
        const geo = new THREE.BufferGeometry().setFromPoints(edge);
        const mat = new THREE.LineBasicMaterial({ color: bnetColor, linewidth: 2 });
        const line = new THREE.Line(geo, mat);
        line.visible = (terrainStyle !== 'satellite'); // hide on satellite by default
        worldGroup.add(line);
        trailBoundaryMeshes.push(line);
    });

    console.log('[race] trail boundary (B-nets) added');
}

function subdivideAndDrape(pts) {
    if (pts.length < 2) return pts;
    const result = [];
    for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        result.push(a.clone());
        for (let t = 1; t <= 2; t++) {
            const f = t / 3;
            const mid = new THREE.Vector3(a.x + (b.x - a.x) * f, 0, a.z + (b.z - a.z) * f);
            const y = getTerrainElevAt(mid.x, mid.z);
            mid.y = (y > 0 ? y : a.y) + 0.5;
            result.push(mid);
        }
    }
    result.push(pts[pts.length - 1].clone());
    return result;
}

// Convert lat/lon to 3D scene coordinates
function geoTo3D(lat, lon, elev) {
    const tm = terrainMeta;
    const dLon = (lon - tm.center.lon) * Math.cos(tm.center.lat * Math.PI / 180) * 111320;
    const dLat = (lat - tm.center.lat) * 110540;
    const y = ((elev || tm.elev_min) - tm.elev_min) * VERT_SCALE * (1 - flattenFactor);
    return new THREE.Vector3(dLon, y + 1, -dLat);
}

function clearCourseOverlay() {
    gateMeshes.forEach(m => worldGroup.remove(m));
    camMeshes.forEach(m => worldGroup.remove(m));
    camFovMeshes.forEach(m => worldGroup.remove(m));
    sectionMeshes.forEach(m => worldGroup.remove(m));
    labelSprites.forEach(m => worldGroup.remove(m));
    terrainInfoSprites.forEach(m => worldGroup.remove(m));
    gateMeshes = []; camMeshes = []; camFovMeshes = []; sectionMeshes = []; labelSprites = [];
    terrainInfoSprites = [];
    gateHoverTargets = [];
    sectionLabelTargets = [];
}

function makeTextSprite(text, opts = {}) {
    const fontSize = opts.fontSize || 28;
    const color = opts.color || '#ffffff';
    const bgColor = opts.bgColor || 'rgba(0,0,0,0.6)';
    const padding = opts.padding || 6;
    const lineHeight = opts.lineHeight || 1.3;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;

    // Support multi-line text
    const lines = text.split('\n');
    const lineWidths = lines.map(l => ctx.measureText(l).width);
    const maxWidth = Math.max(...lineWidths);
    const totalHeight = lines.length * fontSize * lineHeight;

    canvas.width = maxWidth + padding * 2;
    canvas.height = totalHeight + padding * 2;

    ctx.fillStyle = bgColor;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(0, 0, canvas.width, canvas.height, 4);
    else ctx.rect(0, 0, canvas.width, canvas.height);
    ctx.fill();

    ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
    ctx.fillStyle = color;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    lines.forEach((line, idx) => {
        const y = padding + fontSize * lineHeight * (idx + 0.5);
        ctx.fillText(line, canvas.width / 2, y);
    });

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false });
    const sprite = new THREE.Sprite(spriteMat);
    const aspect = canvas.width / canvas.height;
    const spriteScale = opts.scale || 20;
    sprite.scale.set(spriteScale * aspect, spriteScale, 1);
    return sprite;
}

function updateCourseOverlay() {
    clearCourseOverlay();

    const course = manifest.course[activeRun];
    if (!course) return;
    const gates = course.gates;

    // ── GS Gates: two poles + banner panel (like real GS gates) ──
    // Real GS panels: ~75cm wide banner stretched between two bamboo poles ~4m apart.
    // Scaled up for 3D overview visibility. GPS point = internal (turning) pole.
    // Panel orientation: perpendicular to the fall line (cross-slope), so the
    // skier sees the panel face-on when approaching.
    gates.forEach((g, idx) => {
        const elev = g.dem_elev || g.elev;
        const pos = geoTo3D(g.lat, g.lon, elev);
        const isRed = g.color === 'red';
        const color = isRed ? 0xdc2626 : 0x2563eb;
        const poleColor = isRed ? 0xb91c1c : 0x1d4ed8;

        // Fall-line direction: average of vectors to previous and next gate
        let fallX = 0, fallZ = 0;
        if (idx < gates.length - 1) {
            const next = geoTo3D(gates[idx + 1].lat, gates[idx + 1].lon, gates[idx + 1].dem_elev || gates[idx + 1].elev);
            fallX += next.x - pos.x;
            fallZ += next.z - pos.z;
        }
        if (idx > 0) {
            const prev = geoTo3D(gates[idx - 1].lat, gates[idx - 1].lon, gates[idx - 1].dem_elev || gates[idx - 1].elev);
            fallX += pos.x - prev.x;
            fallZ += pos.z - prev.z;
        }
        // Normalize fall line direction
        const fallLen = Math.sqrt(fallX * fallX + fallZ * fallZ) || 1;
        fallX /= fallLen;
        fallZ /= fallLen;

        // Cross-slope perpendicular (rotate fall line 90°)
        const perpX = -fallZ;
        const perpZ = fallX;

        const poleSpacing = 0.6; // ~60cm between poles within one panel gate
        const poleHeight = 3.0; // exaggerated from 1.83m for overview visibility
        const poleRadius = 0.06;

        // Two pole positions: both at same base elevation (turning pole's)
        const pole1X = pos.x;
        const pole1Z = pos.z;
        const baseY = pos.y;
        const pole2X = pos.x + perpX * poleSpacing;
        const pole2Z = pos.z + perpZ * poleSpacing;

        // Pole 1 (internal/turning pole)
        const poleGeo1 = new THREE.CylinderGeometry(poleRadius, poleRadius, poleHeight, 6);
        const poleMat = new THREE.MeshLambertMaterial({ color: poleColor });
        const pole1 = new THREE.Mesh(poleGeo1, poleMat);
        pole1.position.set(pole1X, baseY + poleHeight / 2, pole1Z);
        worldGroup.add(pole1);
        gateMeshes.push(pole1);

        // Pole 2 (external pole) — same height as pole 1
        const poleGeo2 = new THREE.CylinderGeometry(poleRadius, poleRadius, poleHeight, 6);
        const pole2 = new THREE.Mesh(poleGeo2, poleMat);
        pole2.position.set(pole2X, baseY + poleHeight / 2, pole2Z);
        worldGroup.add(pole2);
        gateMeshes.push(pole2);

        // Banner panel stretched between the tops of the two poles
        const panelH = 1.2; // exaggerated from 0.75m for overview visibility
        const poleTop = baseY + poleHeight;
        const panelTopY = poleTop - 0.1;
        const panelCenterY = panelTopY - panelH / 2;

        // Panel center is midpoint between the two pole positions
        const panelCenterX = (pole1X + pole2X) / 2;
        const panelCenterZ = (pole1Z + pole2Z) / 2;

        // Actual distance between poles for panel width
        const actualSpacing = Math.sqrt(
            (pole2X - pole1X) * (pole2X - pole1X) +
            (pole2Z - pole1Z) * (pole2Z - pole1Z)
        );

        // Panel orientation: the panel should be perpendicular to the fall line,
        // meaning its flat face is visible when looking down the slope.
        // PlaneGeometry default normal = +Z, so rotation.y should orient the
        // plane normal along the fall line direction.
        const panelAngle = Math.atan2(fallX, fallZ);

        const panelGeo = new THREE.PlaneGeometry(actualSpacing, panelH);
        const panelMat = new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide });
        const panel = new THREE.Mesh(panelGeo, panelMat);
        panel.position.set(panelCenterX, panelCenterY, panelCenterZ);
        panel.rotation.y = panelAngle;
        worldGroup.add(panel);
        gateMeshes.push(panel);

        // Gate number label floating above the panel — bare colored number, no background
        const gateLabelColor = isRed ? '#dc2626' : '#2563eb';
        const gateLabel = makeTextSprite(String(g.number), {
            bgColor: 'rgba(0,0,0,0)', color: gateLabelColor, scale: 3, fontSize: 20, padding: 2
        });
        gateLabel.position.set(panelCenterX, poleTop + 3, panelCenterZ);
        worldGroup.add(gateLabel);
        labelSprites.push(gateLabel);

        // Hover target — tall capsule covering gate panel + number label above
        const hoverGeo = new THREE.SphereGeometry(3.5, 8, 8);
        const hoverMat = new THREE.MeshBasicMaterial({ visible: false });
        const hoverMesh = new THREE.Mesh(hoverGeo, hoverMat);
        hoverMesh.position.set(panelCenterX, poleTop + 1, panelCenterZ);
        worldGroup.add(hoverMesh);
        gateMeshes.push(hoverMesh);

        // Compute per-gate stats for tooltip
        const mPerLat_h = 110540;
        const mPerLon_h = 111320 * Math.cos(g.lat * Math.PI / 180);
        let tooltipHtml = `<b>Gate ${g.number}</b> (${g.color})`;

        const refLat_h = (idx === 0 && course.start) ? course.start.lat : (idx > 0 ? gates[idx - 1].lat : null);
        const refLon_h = (idx === 0 && course.start) ? course.start.lon : (idx > 0 ? gates[idx - 1].lon : null);
        const refElev_h = (idx === 0 && course.start) ? (course.start.dem_elev || course.start.elev) : (idx > 0 ? (gates[idx - 1].dem_elev || gates[idx - 1].elev) : null);

        if (refLat_h != null) {
            const dLatH = (g.lat - refLat_h) * mPerLat_h;
            const dLonH = (g.lon - refLon_h) * mPerLon_h;
            const dElevH = refElev_h - elev;
            const dist2DH = Math.sqrt(dLatH * dLatH + dLonH * dLonH);
            const dist3DH = Math.sqrt(dist2DH * dist2DH + dElevH * dElevH);
            tooltipHtml += `<br>↕ ${dist3DH.toFixed(1)}m from prev`;

            // Horizontal gate distance (FIS definition) - pre-computed in manifest
            // Perpendicular distance from gate to line between prev and next gates
            const off = g.offset_lr || 0;
            if (Math.abs(off) > 0.01) {
                const dir = off > 0 ? 'right' : 'left';
                tooltipHtml += `<br>${off > 0 ? '→' : '←'} ${Math.abs(off).toFixed(1)}m ${dir}`;
            }

            tooltipHtml += `<br>▼ ${dElevH.toFixed(1)}m drop`;
        }

        if (g.accuracy != null) tooltipHtml += `<br>± ${g.accuracy.toFixed(2)}m GPS`;

        gateHoverTargets.push({
            mesh: hoverMesh,
            label: tooltipHtml
        });

    });

    // ── Start cabin (wooden shack like Proctor start house) ──
    if (course.start) {
        const startPos = geoTo3D(course.start.lat, course.start.lon, course.start.dem_elev || course.start.elev);

        // Determine downhill direction from start toward first gate
        let downX = 0, downZ = 0;
        if (gates.length > 0) {
            const g1 = geoTo3D(gates[0].lat, gates[0].lon, gates[0].dem_elev || gates[0].elev);
            downX = g1.x - startPos.x;
            downZ = g1.z - startPos.z;
        }
        const downLen = Math.sqrt(downX * downX + downZ * downZ) || 1;
        downX /= downLen; downZ /= downLen;
        // Cabin faces downhill; rotation angle for the group
        const cabinAngle = Math.atan2(downX, downZ);

        const cabinGroup = new THREE.Group();
        cabinGroup.position.copy(startPos);
        cabinGroup.rotation.y = cabinAngle;

        const woodDark = 0x6b4226;   // dark brown (walls/structure)
        const woodLight = 0x8b6914;  // lighter brown (trim)
        const roofColor = 0x4a4a4a;  // dark grey shingles
        const deckColor = 0x9e7c4a;  // light wood deck
        const ww = 5, wd = 4, wh = 4; // cabin width, depth, wall height

        // ── Deck/platform in front of cabin ──
        const deckGeo = new THREE.BoxGeometry(ww + 2, 0.3, 3);
        const deckMat = new THREE.MeshLambertMaterial({ color: deckColor });
        const deck = new THREE.Mesh(deckGeo, deckMat);
        deck.position.set(0, 0.15, wd / 2 + 1.5);
        cabinGroup.add(deck);

        // ── Floor ──
        const floorGeo = new THREE.BoxGeometry(ww, 0.2, wd);
        const floorMat = new THREE.MeshLambertMaterial({ color: deckColor });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.position.set(0, 0.1, 0);
        cabinGroup.add(floor);

        // ── Back wall ──
        const backGeo = new THREE.BoxGeometry(ww, wh, 0.25);
        const wallMat = new THREE.MeshLambertMaterial({ color: woodDark });
        const backWall = new THREE.Mesh(backGeo, wallMat);
        backWall.position.set(0, wh / 2, -wd / 2);
        cabinGroup.add(backWall);

        // ── Left wall ──
        const sideGeo = new THREE.BoxGeometry(0.25, wh, wd);
        const leftWall = new THREE.Mesh(sideGeo, wallMat);
        leftWall.position.set(-ww / 2, wh / 2, 0);
        cabinGroup.add(leftWall);

        // ── Right wall ──
        const rightWall = new THREE.Mesh(sideGeo, wallMat);
        rightWall.position.set(ww / 2, wh / 2, 0);
        cabinGroup.add(rightWall);

        // ── Window opening — a darker inset on the back wall (starter's window) ──
        const windowGeo = new THREE.BoxGeometry(2, 1.5, 0.3);
        const windowMat = new THREE.MeshLambertMaterial({ color: 0x1a1a1a });
        const windowMesh = new THREE.Mesh(windowGeo, windowMat);
        windowMesh.position.set(0, wh * 0.65, -wd / 2 + 0.05);
        cabinGroup.add(windowMesh);

        // ── Corner posts (4 vertical timber posts) ──
        const postGeo = new THREE.BoxGeometry(0.4, wh + 1, 0.4);
        const postMat = new THREE.MeshLambertMaterial({ color: woodLight });
        [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx, sz]) => {
            const post = new THREE.Mesh(postGeo, postMat);
            post.position.set(sx * ww / 2, (wh + 1) / 2, sz * wd / 2);
            cabinGroup.add(post);
        });

        // ── Peaked roof (two slanted planes) ──
        const roofMat = new THREE.MeshLambertMaterial({ color: roofColor, side: THREE.DoubleSide });
        const roofOverhang = 0.8;
        const roofRise = 2.0; // peak height above walls
        const roofHalfW = ww / 2 + roofOverhang;
        const roofHalfD = wd / 2 + roofOverhang;

        // Left roof slope
        const lRoofGeo = new THREE.BufferGeometry();
        const lRoofVerts = new Float32Array([
            -roofHalfW, wh, -roofHalfD,
             0, wh + roofRise, -roofHalfD,
            -roofHalfW, wh, roofHalfD,
             0, wh + roofRise, -roofHalfD,
             0, wh + roofRise, roofHalfD,
            -roofHalfW, wh, roofHalfD
        ]);
        lRoofGeo.setAttribute('position', new THREE.BufferAttribute(lRoofVerts, 3));
        lRoofGeo.computeVertexNormals();
        cabinGroup.add(new THREE.Mesh(lRoofGeo, roofMat));

        // Right roof slope
        const rRoofGeo = new THREE.BufferGeometry();
        const rRoofVerts = new Float32Array([
            roofHalfW, wh, -roofHalfD,
            0, wh + roofRise, -roofHalfD,
            roofHalfW, wh, roofHalfD,
            0, wh + roofRise, -roofHalfD,
            0, wh + roofRise, roofHalfD,
            roofHalfW, wh, roofHalfD
        ]);
        rRoofGeo.setAttribute('position', new THREE.BufferAttribute(rRoofVerts, 3));
        rRoofGeo.computeVertexNormals();
        cabinGroup.add(new THREE.Mesh(rRoofGeo, roofMat));

        // ── Gable triangles (front & back) ──
        const gableMat = new THREE.MeshLambertMaterial({ color: woodDark, side: THREE.DoubleSide });
        // Back gable
        const bgGeo = new THREE.BufferGeometry();
        const bgVerts = new Float32Array([
            -ww / 2, wh, -wd / 2,
             ww / 2, wh, -wd / 2,
             0, wh + roofRise, -wd / 2
        ]);
        bgGeo.setAttribute('position', new THREE.BufferAttribute(bgVerts, 3));
        bgGeo.computeVertexNormals();
        cabinGroup.add(new THREE.Mesh(bgGeo, gableMat));

        // Front gable (open side — no lower wall, just the triangle above)
        const fgGeo = new THREE.BufferGeometry();
        const fgVerts = new Float32Array([
            -ww / 2, wh, wd / 2,
             ww / 2, wh, wd / 2,
             0, wh + roofRise, wd / 2
        ]);
        fgGeo.setAttribute('position', new THREE.BufferAttribute(fgVerts, 3));
        fgGeo.computeVertexNormals();
        cabinGroup.add(new THREE.Mesh(fgGeo, gableMat));

        // ── Yellow start wand (timing pole) ──
        const wandGeo = new THREE.CylinderGeometry(0.12, 0.12, 5, 6);
        const wandMat = new THREE.MeshLambertMaterial({ color: 0xfbbf24 });
        const wand = new THREE.Mesh(wandGeo, wandMat);
        wand.position.set(ww / 2 + 1, 2.5, wd / 2 + 1.5);
        cabinGroup.add(wand);

        worldGroup.add(cabinGroup);
        gateMeshes.push(cabinGroup);

        // START label floating above cabin
        const startLabel = makeTextSprite('START', { bgColor: 'rgba(22,163,74,0.85)', color: '#fff', scale: 8 });
        startLabel.position.set(startPos.x, startPos.y + wh + roofRise + 3, startPos.z);
        worldGroup.add(startLabel);
        labelSprites.push(startLabel);
    }

    // ── Finish label + line ──
    let finishPos = null;
    if (course.finish_left && course.finish_right) {
        const fLat = (course.finish_left.lat + course.finish_right.lat) / 2;
        const fLon = (course.finish_left.lon + course.finish_right.lon) / 2;
        finishPos = geoTo3D(fLat, fLon, course.finish_left.dem_elev || course.finish_left.elev);
        const fl = geoTo3D(course.finish_left.lat, course.finish_left.lon, course.finish_left.dem_elev || course.finish_left.elev);
        const fr = geoTo3D(course.finish_right.lat, course.finish_right.lon, course.finish_right.dem_elev || course.finish_right.elev);
        // Blue finish line draped on terrain — thick ribbon mesh
        const finLinePoints = [];
        const finSegs = 16;
        for (let i = 0; i <= finSegs; i++) {
            const t = i / finSegs;
            const px = fl.x + (fr.x - fl.x) * t;
            const pz = fl.z + (fr.z - fl.z) * t;
            const py = getTerrainElevAt(px, pz);
            finLinePoints.push(new THREE.Vector3(px, (py > 0 ? py : fl.y) + 0.5, pz));
        }
        // Build a ribbon: for each point, offset perpendicular to the line direction
        const ribbonWidth = 3.5; // meters wide — visible from overview
        const ribbonVerts = [];
        const ribbonIdx = [];
        for (let i = 0; i < finLinePoints.length; i++) {
            const p = finLinePoints[i];
            let dir;
            if (i < finLinePoints.length - 1) {
                dir = new THREE.Vector3().subVectors(finLinePoints[i + 1], p).normalize();
            } else {
                dir = new THREE.Vector3().subVectors(p, finLinePoints[i - 1]).normalize();
            }
            // Perpendicular in xz plane, offset up slightly
            const perp = new THREE.Vector3(-dir.z, 0, dir.x).normalize();
            ribbonVerts.push(
                p.x + perp.x * ribbonWidth / 2, p.y + 0.3, p.z + perp.z * ribbonWidth / 2,
                p.x - perp.x * ribbonWidth / 2, p.y + 0.3, p.z - perp.z * ribbonWidth / 2
            );
            if (i < finLinePoints.length - 1) {
                const base = i * 2;
                ribbonIdx.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
            }
        }
        const ribbonGeo = new THREE.BufferGeometry();
        ribbonGeo.setAttribute('position', new THREE.Float32BufferAttribute(ribbonVerts, 3));
        ribbonGeo.setIndex(ribbonIdx);
        ribbonGeo.computeVertexNormals();
        const ribbonMat = new THREE.MeshBasicMaterial({ color: 0x3b82f6, side: THREE.DoubleSide }); // bright blue
        const finLine = new THREE.Mesh(ribbonGeo, ribbonMat);
        worldGroup.add(finLine);
        gateMeshes.push(finLine);
    } else if (course.finish_approx) {
        finishPos = geoTo3D(course.finish_approx.lat, course.finish_approx.lon, course.finish_approx.dem_elev || course.finish_approx.elev);
    }
    if (finishPos) {
        const finishLabel = makeTextSprite('FINISH', { bgColor: 'rgba(220,38,38,0.85)', color: '#fff', scale: 8 });
        finishLabel.position.set(finishPos.x + 5, finishPos.y + 6, finishPos.z);
        worldGroup.add(finishLabel);
        labelSprites.push(finishLabel);
    }

    // ── Camera markers + FOV cones (only cameras with coverage for this run) ──
    // Group cameras by position to combine labels for co-located cameras (e.g. Cam 2 & 3)
    const camGroups = {};
    manifest.cameras.forEach(cam => {
        if (!cam.position || !cam.position.lat) return;
        const coverage = getCamCoverage(cam);
        if (coverage.length === 0) return; // no coverage this run

        const posKey = cam.position.lat.toFixed(4) + ',' + cam.position.lon.toFixed(4);
        if (!camGroups[posKey]) camGroups[posKey] = { cams: [], position: cam.position };
        camGroups[posKey].cams.push(cam);
    });

    Object.values(camGroups).forEach(group => {
        const pos0 = group.position;
        const elev = pos0.dem_elev || pos0.elev || terrainMeta.elev_min + 50;
        const pos = geoTo3D(pos0.lat, pos0.lon, elev);

        const camGeo = new THREE.BoxGeometry(1.5, 2.5, 1.5);
        const camMat3d = new THREE.MeshLambertMaterial({ color: 0x0891b2 });
        const camBox = new THREE.Mesh(camGeo, camMat3d);
        camBox.position.copy(pos);
        camBox.position.y += 1.5;
        if (showCameras) worldGroup.add(camBox);
        camMeshes.push(camBox);

        // Label: "Camera" for single, "Cameras" for co-located
        const label = group.cams.length > 1 ? 'Cameras' : 'Camera';
        const camLabel = makeTextSprite(label, { bgColor: 'rgba(8,145,178,0.85)', color: '#fff', scale: 7 });
        camLabel.position.set(pos.x, pos.y + 6, pos.z);
        if (showCameras) worldGroup.add(camLabel);
        camMeshes.push(camLabel);

        // ── FOV cones: one per camera, edges matching section trapezoid ──
        group.cams.forEach(cam => {
            const coverage = getCamCoverage(cam);
            if (coverage.length === 0) return;
            const covGates = gates.filter(g => coverage.includes(g.number));
            if (covGates.length === 0) return;

            const gatePositions = covGates.map(g => geoTo3D(g.lat, g.lon, g.dem_elev || g.elev));

            // Camera position
            const camPosY = pos.y + 2.0;
            const camX = pos.x, camZ = pos.z;

            // Direction from camera to center of covered gates
            let avgGX = 0, avgGZ = 0;
            gatePositions.forEach(p => { avgGX += p.x; avgGZ += p.z; });
            avgGX /= gatePositions.length;
            avgGZ /= gatePositions.length;

            const toGateX = avgGX - camX;
            const toGateZ = avgGZ - camZ;
            const camDist = Math.sqrt(toGateX * toGateX + toGateZ * toGateZ) || 1;
            const dirX = toGateX / camDist;
            const dirZ = toGateZ / camDist;
            const perpX = -dirZ, perpZ = dirX;

            // Project gates to find far edge extent
            let minPerp = Infinity, maxPerp = -Infinity;
            let maxAlong = -Infinity;
            let avgY = 0;
            gatePositions.forEach(p => {
                const dx = p.x - camX;
                const dz = p.z - camZ;
                const along = dx * dirX + dz * dirZ;
                const perp = dx * perpX + dz * perpZ;
                minPerp = Math.min(minPerp, perp);
                maxPerp = Math.max(maxPerp, perp);
                maxAlong = Math.max(maxAlong, along);
                avgY += p.y;
            });
            avgY /= gatePositions.length;

            const pad = 12;
            const farDist = maxAlong + 8;
            const farLeftX = camX + dirX * farDist + perpX * (minPerp - pad);
            const farLeftZ = camZ + dirZ * farDist + perpZ * (minPerp - pad);
            const farRightX = camX + dirX * farDist + perpX * (maxPerp + pad);
            const farRightZ = camZ + dirZ * farDist + perpZ * (maxPerp + pad);

            // Build FOV triangle: camera → far-left → far-right
            const triGeo = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                camX, camPosY, camZ,
                farLeftX, avgY, farLeftZ,
                farRightX, avgY, farRightZ
            ]);
            triGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            triGeo.computeVertexNormals();

            const sectionColor = SECTION_COLORS[cam.id] || 0x888888;
            const triMat = new THREE.MeshBasicMaterial({
                color: sectionColor, transparent: true, opacity: 0.25,
                side: THREE.DoubleSide, depthWrite: false
            });
            const triMesh = new THREE.Mesh(triGeo, triMat);
            if (showCameras) worldGroup.add(triMesh);
            camFovMeshes.push(triMesh);

            // Cone outline
            const outlineGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(camX, camPosY, camZ),
                new THREE.Vector3(farLeftX, avgY, farLeftZ),
                new THREE.Vector3(farRightX, avgY, farRightZ),
                new THREE.Vector3(camX, camPosY, camZ)
            ]);
            const outlineMat = new THREE.LineBasicMaterial({ color: sectionColor, linewidth: 1, transparent: true, opacity: 0.6 });
            const outlineLine = new THREE.Line(outlineGeo, outlineMat);
            if (showCameras) worldGroup.add(outlineLine);
            camFovMeshes.push(outlineLine);
        });
    });

    drawCameraSections(gates);
    addTerrainInfo(gates);
}

/**
 * Draw a trapezoid (FOV footprint) for each camera section in 3D.
 * The trapezoid has its wide edge spanning the covered gates and narrows
 * toward the camera position, with a shaded ground area showing the
 * approximate field of view on the slope.
 * Section numbering is fixed per camera: Cam1=Section 1, Cam2=Section 2, Cam3=Section 3.
 */
function drawCameraSections(gates) {
    manifest.cameras.forEach((cam, camIdx) => {
        const covered = getCamCoverage(cam);
        if (covered.length === 0) return;

        const coveredGates = gates.filter(g => covered.includes(g.number));
        if (coveredGates.length === 0) return;

        if (!cam.position || !cam.position.lat) return;

        const sectionNum = camIdx + 1;
        const sectionColor = SECTION_COLORS[cam.id] || 0x888888;

        // Camera 3D position
        const camElev = cam.position.dem_elev || cam.position.elev || terrainMeta.elev_min + 50;
        const camPos = geoTo3D(cam.position.lat, cam.position.lon, camElev);

        // Covered gate 3D positions
        const gatePositions = coveredGates.map(g => geoTo3D(g.lat, g.lon, g.dem_elev || g.elev));

        // Direction from camera to center of covered gates
        let avgX = 0, avgZ = 0;
        gatePositions.forEach(p => { avgX += p.x; avgZ += p.z; });
        avgX /= gatePositions.length;
        avgZ /= gatePositions.length;

        const camToGateX = avgX - camPos.x;
        const camToGateZ = avgZ - camPos.z;
        const camDist = Math.sqrt(camToGateX * camToGateX + camToGateZ * camToGateZ) || 1;
        const dirX = camToGateX / camDist;
        const dirZ = camToGateZ / camDist;

        // Perpendicular direction (cross-slope relative to camera view)
        const perpX = -dirZ;
        const perpZ = dirX;

        // Project gate positions onto the perpendicular axis to find width
        let minPerp = Infinity, maxPerp = -Infinity;
        // Also find the along-axis extent (distance from camera)
        let minAlong = Infinity, maxAlong = -Infinity;
        gatePositions.forEach(p => {
            const dx = p.x - camPos.x;
            const dz = p.z - camPos.z;
            const along = dx * dirX + dz * dirZ;
            const perp = dx * perpX + dz * perpZ;
            minPerp = Math.min(minPerp, perp);
            maxPerp = Math.max(maxPerp, perp);
            minAlong = Math.min(minAlong, along);
            maxAlong = Math.max(maxAlong, along);
        });

        // Pad the gate span for the far edge (gate side)
        const gatePadding = 12; // meters of padding beyond outermost gates
        const extraLeftPad = cam.id === 'Cam3' ? 6 : 0; // widen left to include gate 21
        const farLeft = minPerp - gatePadding - extraLeftPad;
        const farRight = maxPerp + gatePadding;

        // Far edge: at the furthest gate distance + padding
        const farDist = maxAlong + 8;
        // Near edge: at the closest gate distance - padding (toward camera)
        const nearDist = minAlong - 8;

        // Near edge width narrows proportionally (perspective)
        const narrowFactor = nearDist / farDist;
        const nearLeft = farLeft * narrowFactor;
        const nearRight = farRight * narrowFactor;

        // Build 4 corners of the trapezoid (in 3D space)
        // Near-left, near-right (closer to camera), far-left, far-right (at gates)
        const corners = [
            { x: camPos.x + dirX * nearDist + perpX * nearLeft, z: camPos.z + dirZ * nearDist + perpZ * nearLeft },
            { x: camPos.x + dirX * nearDist + perpX * nearRight, z: camPos.z + dirZ * nearDist + perpZ * nearRight },
            { x: camPos.x + dirX * farDist + perpX * farRight, z: camPos.z + dirZ * farDist + perpZ * farRight },
            { x: camPos.x + dirX * farDist + perpX * farLeft, z: camPos.z + dirZ * farDist + perpZ * farLeft },
        ];

        // Subdivide each edge and drape on terrain for smooth ground following
        const subdivs = 8;
        const trapPts = [];
        for (let side = 0; side < 4; side++) {
            const c1 = corners[side];
            const c2 = corners[(side + 1) % 4];
            for (let i = 0; i < subdivs; i++) {
                const t = i / subdivs;
                const px = c1.x + (c2.x - c1.x) * t;
                const pz = c1.z + (c2.z - c1.z) * t;
                const py = getTerrainElevAt(px, pz);
                trapPts.push(new THREE.Vector3(px, (py > 0 ? py : camPos.y) + 0.8, pz));
            }
        }

        // Shaded ground area — build a mesh from the trapezoid corners draped on terrain
        const gridRes = 10; // subdivisions along each axis for smooth draping
        const groundVerts = [];
        const groundIdx = [];

        // Parameterize: u = 0 (near) to 1 (far), v = 0 (left) to 1 (right)
        for (let iu = 0; iu <= gridRes; iu++) {
            const u = iu / gridRes;
            // Interpolate near edge to far edge
            const leftX = corners[0].x + (corners[3].x - corners[0].x) * u;
            const leftZ = corners[0].z + (corners[3].z - corners[0].z) * u;
            const rightX = corners[1].x + (corners[2].x - corners[1].x) * u;
            const rightZ = corners[1].z + (corners[2].z - corners[1].z) * u;

            for (let iv = 0; iv <= gridRes; iv++) {
                const v = iv / gridRes;
                const px = leftX + (rightX - leftX) * v;
                const pz = leftZ + (rightZ - leftZ) * v;
                const py = getTerrainElevAt(px, pz);
                groundVerts.push(px, (py > 0 ? py : camPos.y) + 0.5, pz);

                if (iu < gridRes && iv < gridRes) {
                    const base = iu * (gridRes + 1) + iv;
                    groundIdx.push(base, base + 1, base + gridRes + 1);
                    groundIdx.push(base + 1, base + gridRes + 2, base + gridRes + 1);
                }
            }
        }

        const groundGeo = new THREE.BufferGeometry();
        groundGeo.setAttribute('position', new THREE.Float32BufferAttribute(groundVerts, 3));
        groundGeo.setIndex(groundIdx);
        groundGeo.computeVertexNormals();
        const groundMat = new THREE.MeshBasicMaterial({
            color: sectionColor, transparent: true, opacity: 0.18,
            side: THREE.DoubleSide, depthWrite: false
        });
        const groundMesh = new THREE.Mesh(groundGeo, groundMat);
        worldGroup.add(groundMesh);
        sectionMeshes.push(groundMesh);

        // Trapezoid outline (dashed)
        const outlinePts = trapPts.concat([trapPts[0].clone()]);
        const dashLen = 4, gapLen = 3;
        const dashPts = [];
        for (let i = 0; i < outlinePts.length - 1; i++) {
            const a = outlinePts[i], b = outlinePts[i + 1];
            const segLen = a.distanceTo(b);
            if (segLen < 0.1) continue;
            const segDir = new THREE.Vector3().subVectors(b, a).normalize();
            let d = 0, drawing = true;
            while (d < segLen) {
                const end = Math.min(d + (drawing ? dashLen : gapLen), segLen);
                if (drawing) {
                    dashPts.push(
                        new THREE.Vector3().copy(a).addScaledVector(segDir, d),
                        new THREE.Vector3().copy(a).addScaledVector(segDir, end)
                    );
                }
                d = end;
                drawing = !drawing;
            }
        }

        if (dashPts.length >= 2) {
            const geo = new THREE.BufferGeometry().setFromPoints(dashPts);
            const mat = new THREE.LineBasicMaterial({ color: sectionColor, linewidth: 2, transparent: true, opacity: 0.7 });
            const line = new THREE.LineSegments(geo, mat);
            worldGroup.add(line);
            sectionMeshes.push(line);
        }

        // Section label as floating sprite (clickable for trigger zone popup)
        const midGate = coveredGates[Math.floor(coveredGates.length / 2)];
        const midPos = geoTo3D(midGate.lat, midGate.lon, midGate.dem_elev || midGate.elev);

        const hexColor = SECTION_COLORS_CSS[cam.id] || '#888888';
        const r = parseInt(hexColor.slice(1,3), 16) || 0;
        const gv = parseInt(hexColor.slice(3,5), 16) || 0;
        const b = parseInt(hexColor.slice(5,7), 16) || 0;
        const sectionBg = `rgba(${r},${gv},${b},0.8)`;
        const hasZoneImg = cam.trigger_zone_images || cam.trigger_zone_image;
        const labelText = hasZoneImg ? '\u{1F4F7} Section ' + sectionNum : 'Section ' + sectionNum;
        const sectionLabel = makeTextSprite(labelText, {
            bgColor: sectionBg, color: '#fff', scale: 7, fontSize: 24
        });
        sectionLabel.position.set(midPos.x + 12, midPos.y + 6, midPos.z);
        worldGroup.add(sectionLabel);
        labelSprites.push(sectionLabel);

        // Register clickable target for trigger zone popup
        if (hasZoneImg) {
            sectionLabelTargets.push({ sprite: sectionLabel, camId: cam.id });
        }
    });
}


/**
 * Add terrain measurement info (toggled via Measurements checkbox):
 * - Elevation labels along the trail boundary (B-net)
 * Per-gate stats are shown via hover tooltip on gate markers.
 */
function addTerrainInfo(gates) {
    // Clear previous terrain info sprites
    terrainInfoSprites.forEach(m => worldGroup.remove(m));
    terrainInfoSprites = [];

    const course = manifest.course[activeRun];
    if (!course || !gates || gates.length < 2) return;

    // ── Elevation labels along the B-net (trail boundary) ──
    // Place elevation labels at every 3rd gate position, offset to the right side
    const allPts = [];
    if (course.start) allPts.push(course.start);
    gates.forEach(g => allPts.push(g));

    for (let i = 0; i < allPts.length; i += 3) {
        const pt = allPts[i];
        const elev = pt.dem_elev || pt.elev;
        const pos = geoTo3D(pt.lat, pt.lon, elev);

        // Compute perpendicular offset for placement near B-net
        let dirX = 0, dirZ = 0;
        if (i < allPts.length - 1) {
            const next = geoTo3D(allPts[i + 1].lat, allPts[i + 1].lon, allPts[i + 1].dem_elev || allPts[i + 1].elev);
            dirX = next.x - pos.x;
            dirZ = next.z - pos.z;
        } else if (i > 0) {
            const prev = geoTo3D(allPts[i - 1].lat, allPts[i - 1].lon, allPts[i - 1].dem_elev || allPts[i - 1].elev);
            dirX = pos.x - prev.x;
            dirZ = pos.z - prev.z;
        }
        const len = Math.sqrt(dirX * dirX + dirZ * dirZ) || 1;
        const perpX = -(dirZ / len);
        const perpZ = (dirX / len);

        const elevLabel = makeTextSprite(Math.round(elev) + 'm', {
            bgColor: 'rgba(0,0,0,0.55)', color: '#fff', scale: 3, fontSize: 16, padding: 3
        });
        elevLabel.position.set(pos.x + perpX * 22, pos.y + 1, pos.z + perpZ * 22);
        worldGroup.add(elevLabel);
        terrainInfoSprites.push(elevLabel);
    }

    console.log('[race] terrain info labels added:', terrainInfoSprites.length);
}

// ── Camera visibility toggles ───────────────────────────────────────────

function toggleCameras3D() {
    camMeshes.forEach(m => {
        if (showCameras) worldGroup.add(m);
        else worldGroup.remove(m);
    });
    camFovMeshes.forEach(m => {
        if (showCameras) worldGroup.add(m);
        else worldGroup.remove(m);
    });
}

function toggleCameras2D() {
    if (!leafletMap) return;
    leafletLayers.cameras.forEach(l => {
        if (showCameras) l.addTo(leafletMap);
        else leafletMap.removeLayer(l);
    });
    if (leafletLayers.camFov) {
        leafletLayers.camFov.forEach(l => {
            if (showCameras) l.addTo(leafletMap);
            else leafletMap.removeLayer(l);
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════
// LEAFLET 2D MAP
// ══════════════════════════════════════════════════════════════════════════

let leafletLayers = { gates: [], cameras: [], sections: [], camFov: [], measurements: [] };

function initLeaflet() {
    const course = manifest.course.run2;
    const allLats = course.gates.map(g => g.lat);
    const allLons = course.gates.map(g => g.lon);
    if (course.start) { allLats.push(course.start.lat); allLons.push(course.start.lon); }
    if (course.finish_left) { allLats.push(course.finish_left.lat); allLons.push(course.finish_left.lon); }
    if (course.finish_right) { allLats.push(course.finish_right.lat); allLons.push(course.finish_right.lon); }

    const cLat = allLats.reduce((a, b) => a + b, 0) / allLats.length;
    const cLon = allLons.reduce((a, b) => a + b, 0) / allLons.length;

    // South-up orientation: bearing 180° so start (south) is at top, finish (north) at bottom
    leafletMap = L.map('map', {
        rotate: true, bearing: 180,
        touchRotate: false,
        rotateControl: false,
        keyboard: true,
        maxZoom: 19
    }).setView([cLat, cLon], 16);

    // Auto-focus map so keyboard controls work immediately
    leafletMap.whenReady(() => {
        leafletMap.getContainer().focus();
    });

    const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 20, attribution: 'Esri'
    });
    const topo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        maxZoom: 19, attribution: 'OpenTopoMap'
    });
    sat.addTo(leafletMap);

    L.control.layers(
        { 'Satellite': sat, 'Topo contours': topo },
        {},
        { collapsed: false }
    ).addTo(leafletMap);
    L.control.scale({ imperial: true, metric: true }).addTo(leafletMap);

    updateLeafletOverlay();

    document.getElementById('run-select-2d').addEventListener('change', (e) => {
        activeRun = e.target.value;
        document.getElementById('run-select-3d').value = activeRun;
        updateLeafletOverlay();
        if (scene) updateCourseOverlay();
        syncRunTabs();
    });

    // Camera toggle checkbox (2D)
    document.getElementById('show-cameras-2d').addEventListener('change', (e) => {
        showCameras = e.target.checked;
        // Sync 3D checkbox
        const cb3d = document.getElementById('show-cameras-3d');
        if (cb3d) cb3d.checked = showCameras;
        toggleCameras2D();
        if (scene) toggleCameras3D();
    });

    // Measurements toggle checkbox (2D)
    document.getElementById('show-measurements-2d').addEventListener('change', (e) => {
        showMeasurements = e.target.checked;
        toggleMeasurements2D();
    });

    // Fix leaflet-rotate marker drift: force re-render after zoom ends
    leafletMap.on('zoomend', () => {
        // Nudge all divIcon markers to force position recalculation
        const allLayers = [
            ...leafletLayers.gates,
            ...leafletLayers.cameras,
            ...leafletLayers.measurements
        ];
        allLayers.forEach(l => {
            if (l instanceof L.Marker && leafletMap.hasLayer(l)) {
                const pos = l.getLatLng();
                l.setLatLng(pos);
            }
        });
    });

    // Fit bounds with all course points including start and finish
    const bounds = [];
    if (course.start) bounds.push([course.start.lat, course.start.lon]);
    course.gates.forEach(g => bounds.push([g.lat, g.lon]));
    if (course.finish_left) bounds.push([course.finish_left.lat, course.finish_left.lon]);
    if (course.finish_right) bounds.push([course.finish_right.lat, course.finish_right.lon]);
    leafletMap.fitBounds(bounds, { padding: [40, 40], maxZoom: 16 });
}

function clearLeafletOverlay() {
    leafletLayers.gates.forEach(l => leafletMap.removeLayer(l));
    leafletLayers.cameras.forEach(l => leafletMap.removeLayer(l));
    leafletLayers.sections.forEach(l => leafletMap.removeLayer(l));
    if (leafletLayers.camFov) leafletLayers.camFov.forEach(l => leafletMap.removeLayer(l));
    if (leafletLayers.measurements) leafletLayers.measurements.forEach(l => leafletMap.removeLayer(l));
    leafletLayers = { gates: [], cameras: [], sections: [], camFov: [], measurements: [] };
}

function updateLeafletOverlay() {
    if (!leafletMap) return;
    clearLeafletOverlay();

    const course = manifest.course[activeRun];
    if (!course) return;

    // ── Start marker ──
    if (course.start) {
        const m = L.marker([course.start.lat, course.start.lon], {
            icon: L.divIcon({
                className: 'gate-label-2d',
                html: '<span style="color:#16a34a;font-size:14px;font-weight:900;">&#9650; START</span>',
                iconSize: [80, 18], iconAnchor: [10, 9]
            })
        }).addTo(leafletMap);
        leafletLayers.gates.push(m);
    }

    // ── Finish marker ──
    if (course.finish_left && course.finish_right) {
        const fLat = (course.finish_left.lat + course.finish_right.lat) / 2;
        const fLon = (course.finish_left.lon + course.finish_right.lon) / 2;
        const m = L.marker([fLat, fLon], {
            icon: L.divIcon({
                className: 'gate-label-2d',
                html: '<span style="color:#dc2626;font-size:14px;font-weight:900;">&#9632; FINISH</span>',
                iconSize: [80, 18], iconAnchor: [10, 9]
            })
        }).addTo(leafletMap);
        leafletLayers.gates.push(m);
        leafletLayers.gates.push(L.polyline(
            [[course.finish_left.lat, course.finish_left.lon], [course.finish_right.lat, course.finish_right.lon]],
            { color: '#3b82f6', weight: 5 }
        ).addTo(leafletMap));
    }

    // ── Gate markers: panel gates (two poles + line) ──
    const gateSpacingM = 0.6; // ~60cm between poles within one panel gate
    const metersPerLat = 110540;
    const metersPerLon = 111320 * Math.cos((course.gates[0]?.lat || 43.43) * Math.PI / 180);

    course.gates.forEach((g, idx) => {
        const color = g.color === 'red' ? '#dc2626' : '#2563eb';
        const gates = course.gates;

        // Fall-line direction in lat/lon
        let fallLat = 0, fallLon = 0;
        if (idx < gates.length - 1) {
            fallLat += gates[idx + 1].lat - g.lat;
            fallLon += gates[idx + 1].lon - g.lon;
        }
        if (idx > 0) {
            fallLat += g.lat - gates[idx - 1].lat;
            fallLon += g.lon - gates[idx - 1].lon;
        }
        // Normalize in meters then back to degrees
        const fLenM = Math.sqrt((fallLat * metersPerLat) ** 2 + (fallLon * metersPerLon) ** 2) || 1;
        const fLatN = (fallLat * metersPerLat) / fLenM;
        const fLonN = (fallLon * metersPerLon) / fLenM;

        // Perpendicular (cross-slope): rotate 90°
        const perpLatM = -fLonN; // in meters
        const perpLonM = fLatN;

        // Second pole position (offset by gateSpacingM)
        const pole2Lat = g.lat + (perpLatM * gateSpacingM) / metersPerLat;
        const pole2Lon = g.lon + (perpLonM * gateSpacingM) / metersPerLon;

        // Pole 1 (turning pole at GPS position)
        const p1 = L.circleMarker([g.lat, g.lon], {
            radius: 3, fillColor: color, color: color, weight: 1, fillOpacity: 0.9
        }).addTo(leafletMap).bindPopup(`<b>Gate ${g.number}</b> (${g.color})<br>Elev: ${Math.round(g.dem_elev || g.elev)}m`);
        leafletLayers.gates.push(p1);

        // Pole 2 (outside pole)
        const p2 = L.circleMarker([pole2Lat, pole2Lon], {
            radius: 3, fillColor: color, color: color, weight: 1, fillOpacity: 0.9
        }).addTo(leafletMap);
        leafletLayers.gates.push(p2);

        // Panel line between poles
        const panelLine = L.polyline([[g.lat, g.lon], [pole2Lat, pole2Lon]], {
            color: color, weight: 3, opacity: 0.9
        }).addTo(leafletMap);
        leafletLayers.gates.push(panelLine);

        // Gate number label
        leafletLayers.gates.push(L.marker([g.lat, g.lon], {
            icon: L.divIcon({ className: 'gate-label-2d', html: `<span style="color:${color}">${g.number}</span>`, iconSize: [20, 14], iconAnchor: [-8, 7] })
        }).addTo(leafletMap));
    });

    // ── Camera markers + FOV cones (grouped by position for co-located cameras) ──
    const camGroups2D = {};
    manifest.cameras.forEach(cam => {
        if (!cam.position || !cam.position.lat) return;
        const coverage = getCamCoverage(cam);
        if (coverage.length === 0) return;
        const posKey = cam.position.lat.toFixed(4) + ',' + cam.position.lon.toFixed(4);
        if (!camGroups2D[posKey]) camGroups2D[posKey] = { cams: [], position: cam.position };
        camGroups2D[posKey].cams.push(cam);
    });

    Object.values(camGroups2D).forEach(group => {
        const pos = [group.position.lat, group.position.lon];
        const label = group.cams.length > 1 ? 'Cameras' : 'Camera';
        const notes = group.cams.map(c => c.note || '').filter(Boolean).join('; ');

        // Camera icon pointing south (uphill) — FOV cones show exact coverage direction
        const camMarker = L.marker(pos, {
            icon: L.divIcon({
                className: 'cam-label-2d',
                html: `<span style="display:inline-flex;align-items:center;gap:3px;font-size:13px;font-weight:700;color:#0891b2;">` +
                      `<span style="font-size:16px;">&#127909;</span> ${label}</span>`,
                iconSize: [110, 24], iconAnchor: [12, 12]
            })
        }).bindPopup(`<b>${label}</b> (pointing south/uphill)<br>${notes}`);
        if (showCameras) camMarker.addTo(leafletMap);
        leafletLayers.cameras.push(camMarker);

        // ── FOV cones in 2D: triangle from camera to far edge of section trapezoid ──
        group.cams.forEach(cam => {
            const coverage = getCamCoverage(cam);
            if (coverage.length === 0) return;
            const covGates = course.gates.filter(g => coverage.includes(g.number));
            if (covGates.length === 0) return;

            const camLat = group.position.lat, camLon = group.position.lon;
            const mPerLat = 110540;
            const mPerLon = 111320 * Math.cos(camLat * Math.PI / 180);

            // Direction from camera to center of covered gates
            const avgLat = covGates.reduce((s, g) => s + g.lat, 0) / covGates.length;
            const avgLon = covGates.reduce((s, g) => s + g.lon, 0) / covGates.length;
            const dx = (avgLat - camLat) * mPerLat;
            const dy = (avgLon - camLon) * mPerLon;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const dirLat = dx / dist, dirLon = dy / dist;
            const perpLat = -dirLon, perpLon = dirLat;

            // Project gates onto perpendicular + along axes
            let minPerp = Infinity, maxPerp = -Infinity, maxAlong = -Infinity;
            covGates.forEach(g => {
                const gx = (g.lat - camLat) * mPerLat;
                const gy = (g.lon - camLon) * mPerLon;
                const along = gx * dirLat + gy * dirLon;
                const perp = gx * perpLat + gy * perpLon;
                minPerp = Math.min(minPerp, perp);
                maxPerp = Math.max(maxPerp, perp);
                maxAlong = Math.max(maxAlong, along);
            });

            const pad = 12;
            const farDist = maxAlong + 8;
            const farLeft = minPerp - pad;
            const farRight = maxPerp + pad;

            // Far corners of the FOV
            const farLeftLat = camLat + (dirLat * farDist + perpLat * farLeft) / mPerLat;
            const farLeftLon = camLon + (dirLon * farDist + perpLon * farLeft) / mPerLon;
            const farRightLat = camLat + (dirLat * farDist + perpLat * farRight) / mPerLat;
            const farRightLon = camLon + (dirLon * farDist + perpLon * farRight) / mPerLon;

            const sectionColor = SECTION_COLORS_CSS[cam.id] || '#888';
            const fovTriangle = L.polygon([
                [camLat, camLon],
                [farLeftLat, farLeftLon],
                [farRightLat, farRightLon]
            ], {
                color: sectionColor, weight: 1, opacity: 0.6,
                fillColor: sectionColor, fillOpacity: 0.15
            });
            if (showCameras) fovTriangle.addTo(leafletMap);
            leafletLayers.camFov.push(fovTriangle);
        });
    });

    // ── Camera section overlays (per-run) — trapezoid FOV footprints, fixed numbering ──
    manifest.cameras.forEach((cam, camIdx) => {
        if (!cam.position || !cam.position.lat) return;
        const coverage = getCamCoverage(cam);
        if (coverage.length === 0) return;

        const sectionNum = camIdx + 1; // Fixed: Cam1=1, Cam2=2, Cam3=3

        const covGates = course.gates.filter(g => coverage.includes(g.number));
        if (covGates.length >= 1) {
            const sectionColor = SECTION_COLORS_CSS[cam.id] || '#888';

            const camLat = cam.position.lat;
            const camLon = cam.position.lon;

            // Direction from camera to center of covered gates (in meter space)
            const gateLats = covGates.map(g => g.lat);
            const gateLons = covGates.map(g => g.lon);
            const avgLat = gateLats.reduce((a, b) => a + b, 0) / gateLats.length;
            const avgLon = gateLons.reduce((a, b) => a + b, 0) / gateLons.length;

            const mPerLat = 110540;
            const mPerLon = 111320 * Math.cos(camLat * Math.PI / 180);

            const camToGateLat = (avgLat - camLat) * mPerLat;
            const camToGateLon = (avgLon - camLon) * mPerLon;
            const camDist = Math.sqrt(camToGateLat ** 2 + camToGateLon ** 2) || 1;
            const dirLat = camToGateLat / camDist;
            const dirLon = camToGateLon / camDist;

            // Perpendicular (cross-view)
            const perpLat = -dirLon;
            const perpLon = dirLat;

            // Project gate positions onto the perpendicular and along axes
            let minPerp = Infinity, maxPerp = -Infinity;
            let minAlong = Infinity, maxAlong = -Infinity;
            covGates.forEach(g => {
                const dx = (g.lat - camLat) * mPerLat;
                const dy = (g.lon - camLon) * mPerLon;
                const along = dx * dirLat + dy * dirLon;
                const perp = dx * perpLat + dy * perpLon;
                minPerp = Math.min(minPerp, perp);
                maxPerp = Math.max(maxPerp, perp);
                minAlong = Math.min(minAlong, along);
                maxAlong = Math.max(maxAlong, along);
            });

            // Pad the gate span
            const gatePadding = 12; // meters
            const extraLeftPad = cam.id === 'Cam3' ? 6 : 0; // widen left to include gate 21
            const farLeft = minPerp - gatePadding - extraLeftPad;
            const farRight = maxPerp + gatePadding;
            const farDist = maxAlong + 8;
            const nearDist = minAlong - 8;

            // Near edge narrows proportionally (perspective)
            const narrowFactor = nearDist / farDist;
            const nearLeft = farLeft * narrowFactor;
            const nearRight = farRight * narrowFactor;

            // Convert back to lat/lon: camera + offset in meters
            function mToLatLon(alongM, perpM) {
                const lat = camLat + (dirLat * alongM + perpLat * perpM) / mPerLat;
                const lon = camLon + (dirLon * alongM + perpLon * perpM) / mPerLon;
                return [lat, lon];
            }

            const trapCorners = [
                mToLatLon(nearDist, nearLeft),   // near-left
                mToLatLon(nearDist, nearRight),  // near-right
                mToLatLon(farDist, farRight),    // far-right
                mToLatLon(farDist, farLeft),     // far-left
            ];

            // Shaded trapezoid polygon (clickable for trigger zone popup)
            const sectionPoly = L.polygon(trapCorners, {
                color: sectionColor, weight: 2, opacity: 0.7,
                dashArray: '8,6',
                fillColor: sectionColor, fillOpacity: 0.12
            }).addTo(leafletMap);
            const hasZoneImg2d = cam.trigger_zone_images || cam.trigger_zone_image;
            if (hasZoneImg2d) {
                sectionPoly.on('click', () => showSectionPopup(cam.id));
                // Add pointer cursor to SVG path
                sectionPoly.on('add', () => {
                    const el = sectionPoly.getElement();
                    if (el) el.style.cursor = 'pointer';
                });
            }
            leafletLayers.sections.push(sectionPoly);

            // Section label (clickable for trigger zone popup)
            const midGate = covGates[Math.floor(covGates.length / 2)];
            const labelIcon = hasZoneImg2d ? '\u{1F4F7} ' : '';
            const sectionMarker = L.marker([midGate.lat, midGate.lon], {
                icon: L.divIcon({
                    className: 'cam-label-2d',
                    html: `<span style="color:${sectionColor};font-size:11px;cursor:${hasZoneImg2d ? 'pointer' : 'default'}">${labelIcon}Section ${sectionNum}</span>`,
                    iconSize: [85, 16], iconAnchor: [-12, 8]
                })
            }).addTo(leafletMap);
            if (hasZoneImg2d) {
                sectionMarker.on('click', () => showSectionPopup(cam.id));
            }
            leafletLayers.sections.push(sectionMarker);
        }
    });

    // ── Per-gate measurements + elevation labels (2D) ──
    addLeafletMeasurements(course);
}

/**
 * Add per-gate measurement labels and elevation markers to the 2D Leaflet map.
 */
function addLeafletMeasurements(course) {
    if (!course || !course.gates || course.gates.length < 2) return;
    const gates = course.gates;

    const mPerLat = 110540;
    const mPerLon = 111320 * Math.cos((gates[0].lat) * Math.PI / 180);

    // ── Elevation labels every 3 gates ──
    const allPts = [];
    if (course.start) allPts.push(course.start);
    gates.forEach(g => allPts.push(g));

    for (let i = 0; i < allPts.length; i += 3) {
        const pt = allPts[i];
        const elev = pt.dem_elev || pt.elev;

        // Offset position slightly to the right for placement
        let fallLat = 0, fallLon = 0;
        if (i < allPts.length - 1) {
            fallLat = allPts[i + 1].lat - pt.lat;
            fallLon = allPts[i + 1].lon - pt.lon;
        } else if (i > 0) {
            fallLat = pt.lat - allPts[i - 1].lat;
            fallLon = pt.lon - allPts[i - 1].lon;
        }
        const fLen = Math.sqrt((fallLat * mPerLat) ** 2 + (fallLon * mPerLon) ** 2) || 1;
        // Perpendicular offset (~20m to the right)
        const perpLatM = -(fallLon * mPerLon) / fLen;
        const perpLonM = (fallLat * mPerLat) / fLen;
        const offsetM = 20;
        const labelLat = pt.lat + (perpLatM * offsetM) / mPerLat;
        const labelLon = pt.lon + (perpLonM * offsetM) / mPerLon;

        const m = L.marker([labelLat, labelLon], {
            icon: L.divIcon({
                className: 'gate-label-2d',
                html: `<span style="color:#1e293b;font-size:10px;font-weight:600;background:rgba(255,255,255,0.7);padding:1px 3px;border-radius:2px;">${Math.round(elev)}m</span>`,
                iconSize: [36, 14], iconAnchor: [18, 7]
            })
        });
        if (showMeasurements) m.addTo(leafletMap);
        leafletLayers.measurements.push(m);
    }

    // ── Per-gate stats ──
    for (let i = 0; i < gates.length; i++) {
        const g = gates[i];
        const elev = g.dem_elev || g.elev;

        const refLat = (i === 0 && course.start) ? course.start.lat : (i > 0 ? gates[i - 1].lat : g.lat);
        const refLon = (i === 0 && course.start) ? course.start.lon : (i > 0 ? gates[i - 1].lon : g.lon);
        const refElev = (i === 0 && course.start) ? (course.start.dem_elev || course.start.elev) : (i > 0 ? (gates[i - 1].dem_elev || gates[i - 1].elev) : elev);

        if (i === 0 && !course.start) continue; // no previous reference

        const dLat = (g.lat - refLat) * mPerLat;
        const dLon = (g.lon - refLon) * mPerLon;
        const dElev = refElev - elev;
        const dist2D = Math.sqrt(dLat * dLat + dLon * dLon);
        const dist3D = Math.sqrt(dist2D * dist2D + dElev * dElev);

        // Horizontal gate distance (FIS definition) - pre-computed in manifest
        // Perpendicular distance from gate to line between prev and next gates
        const offset = g.offset_lr || 0;

        const accuracy = g.accuracy;

        // Build compact label
        const parts = [];
        parts.push(`↕${dist3D.toFixed(1)}`);
        if (Math.abs(offset) > 0.01) {
            const dir = offset > 0 ? '→' : '←';
            parts.push(`${dir}${Math.abs(offset).toFixed(1)}`);
        }
        parts.push(`▼${dElev.toFixed(1)}`);
        if (accuracy != null) parts.push(`±${accuracy.toFixed(1)}`);

        const labelText = parts.join(' ');

        // Position: offset to the left of the gate
        let fallLat = 0, fallLon = 0;
        if (i < gates.length - 1) {
            fallLat += gates[i + 1].lat - g.lat;
            fallLon += gates[i + 1].lon - g.lon;
        }
        if (i > 0) {
            fallLat += g.lat - gates[i - 1].lat;
            fallLon += g.lon - gates[i - 1].lon;
        }
        const fLen = Math.sqrt((fallLat * mPerLat) ** 2 + (fallLon * mPerLon) ** 2) || 1;
        const perpLatM = (fallLon * mPerLon) / fLen;
        const perpLonM = -(fallLat * mPerLat) / fLen;
        const offsetDist = 5; // meters offset
        const labelLat = g.lat + (perpLatM * offsetDist) / mPerLat;
        const labelLon = g.lon + (perpLonM * offsetDist) / mPerLon;

        const m = L.marker([labelLat, labelLon], {
            icon: L.divIcon({
                className: 'measure-label-2d',
                html: `<span>${labelText}</span>`,
                iconSize: [120, 14], iconAnchor: [-4, 7]
            })
        });
        if (showMeasurements) m.addTo(leafletMap);
        leafletLayers.measurements.push(m);
    }
}

function toggleMeasurements2D() {
    if (!leafletMap) return;
    leafletLayers.measurements.forEach(l => {
        if (showMeasurements) l.addTo(leafletMap);
        else leafletMap.removeLayer(l);
    });
    // Toggle measurement legend visibility
    const legendEl = document.getElementById('legend-measurements-2d');
    if (legendEl) legendEl.style.display = showMeasurements ? 'block' : 'none';
}


// ══════════════════════════════════════════════════════════════════════════
// RESULTS TABLE — Per-run, fully independent
// ══════════════════════════════════════════════════════════════════════════

/**
 * Get time and status for an athlete for the active run.
 */
function getRunTime(a) {
    if (activeRun === 'run1') return { time: a.run1_time, status: a.run1_status };
    return { time: a.run2_time, status: a.run2_status };
}

/**
 * Compute ranking for the active run only.
 * Only athletes who finished the active run get a rank.
 */
function computeRunRanks(athletes) {
    const finished = athletes
        .filter(a => {
            const { time, status } = getRunTime(a);
            return status === 'finished' && time != null;
        })
        .sort((a, b) => getRunTime(a).time - getRunTime(b).time);

    const ranks = {};
    finished.forEach((a, i) => { ranks[a.bib] = i + 1; });
    return ranks;
}

function formatTimeCell(time, status) {
    if (status === 'DSQ') return '<span class="run-dsq">DSQ</span>';
    if (status === 'DNF') return '<span class="run-dnf">DNF</span>';
    if (status === 'DNS') return '<span class="run-dns">DNS</span>';
    if (time != null) return `<span class="run-val">${time.toFixed(2)}</span>`;
    return '';
}

/**
 * Get which cameras have coverage for the active run.
 * Returns array of {cam, sectionIdx} for cameras active this run.
 * Section numbering is fixed: Cam1=1, Cam2=2, Cam3=3 (not sequential).
 */
function getActiveSections() {
    const sections = [];
    manifest.cameras.forEach((cam, camIdx) => {
        const coverage = getCamCoverage(cam);
        if (coverage.length > 0) {
            sections.push({ cam, sectionIdx: camIdx + 1 });
        }
    });
    return sections;
}

/**
 * Get section elapsed time for an athlete/camera/run.
 * Uses first detection's section_time from the array.
 * Returns number (seconds) or null.
 */
function getSectionTime(athlete, camId, runKey) {
    if (!athlete.montages || !athlete.montages[camId] || !athlete.montages[camId][runKey]) return null;
    const dets = athlete.montages[camId][runKey];
    if (!Array.isArray(dets) || dets.length === 0) return null;
    return dets[0].section_time || null;
}

window.handleSort = function handleSort(column) {
    if (sortColumn === column) {
        // Cycle: asc → desc → off
        if (sortDirection === 'asc') sortDirection = 'desc';
        else { sortColumn = ''; sortDirection = ''; }
    } else {
        sortColumn = column;
        sortDirection = 'asc';
    }
    renderResults();
}

function renderResults() {
    const cat = manifest.categories.find(c => c.id === activeCategory);
    if (!cat) return;

    const tbody = document.getElementById('results-body');
    const thead = document.getElementById('results-thead');

    // Get sections active for this run
    const activeSections = getActiveSections();

    // Build table header with two rows: section labels, column headers
    const sectionColCount = activeSections.length * 4; // time + PM + V + AI per section
    let headerHtml = '<tr>';
    // Row 1: Empty group headers + Per-section sub-headers (clickable → focuses 3D map on that section)
    headerHtml += `<th class="th-group" colspan="2" rowspan="1"></th>`;
    headerHtml += `<th class="th-group" colspan="3" rowspan="1"></th>`;
    activeSections.forEach(({ cam, sectionIdx }) => {
        const icons = '<span class="section-nav-icons">'
            + `<span class="section-nav-btn" onclick="event.stopPropagation(); focusOnSection('${cam.id}')" title="View in 3D map">\u{1F30D}</span>`
            + `<span class="section-nav-btn" onclick="event.stopPropagation(); focusOnSection2D('${cam.id}')" title="View in 2D map">\u{1F9ED}</span>`
            + ((cam.trigger_zone_images || cam.trigger_zone_image)
                ? `<span class="section-nav-btn" onclick="event.stopPropagation(); showSectionPopup('${cam.id}')" title="View trigger zones">\u{1F4F7}</span>`
                : '')
            + '</span>';
        headerHtml += `<th class="th-section-group th-section-clickable" colspan="4" onclick="focusOnSection('${cam.id}')" title="Click to view Section ${sectionIdx} in 3D map">Section ${sectionIdx} ${icons}</th>`;
    });
    headerHtml += '</tr><tr>';
    // Row 3: Column headers
    const rankSortCls = sortColumn === 'rank' ? (sortDirection === 'asc' ? 'sort-asc' : 'sort-desc') : '';
    const timeSortCls = sortColumn === 'time' ? (sortDirection === 'asc' ? 'sort-asc' : 'sort-desc') : '';
    const bibSortCls = sortColumn === 'bib' ? (sortDirection === 'asc' ? 'sort-asc' : 'sort-desc') : '';
    headerHtml += `<th class="col-rank sortable ${rankSortCls}" onclick="handleSort('rank')">Run Rank <span class="sort-arrow">${sortColumn === 'rank' ? (sortDirection === 'asc' ? '\u25B2' : '\u25BC') : '\u25B2'}</span></th>`;
    headerHtml += `<th class="col-time sortable ${timeSortCls}" onclick="handleSort('time')">Time <span class="sort-arrow">${sortColumn === 'time' ? (sortDirection === 'asc' ? '\u25B2' : '\u25BC') : '\u25B2'}</span></th>`;
    headerHtml += `<th class="col-bib sortable ${bibSortCls}" onclick="handleSort('bib')">Bib <span class="sort-arrow">${sortColumn === 'bib' ? (sortDirection === 'asc' ? '\u25B2' : '\u25BC') : '\u25B2'}</span></th>`;
    headerHtml += `<th class="col-name">Name</th>`;
    headerHtml += `<th class="col-club">Club</th>`;
    const sectionTimeDisclaimer = 'Unofficial estimate from camera analysis, not official timing. For coaching use only. Accuracy ±0.08s';
    activeSections.forEach(({ cam, sectionIdx }) => {
        const colKey = 'sectionTime_' + cam.id;
        const stSortCls = sortColumn === colKey ? (sortDirection === 'asc' ? 'sort-asc' : 'sort-desc') : '';
        headerHtml += `<th class="col-section-time sortable ${stSortCls}" onclick="handleSort('${colKey}')" title="${sectionTimeDisclaimer}"><a href="#disclaimer" class="section-time-link" onclick="event.stopPropagation(); document.getElementById('disclaimer').scrollIntoView({behavior:'smooth'}); return false;">Time*</a> <span class="sort-arrow">${sortColumn === colKey ? (sortDirection === 'asc' ? '\u25B2' : '\u25BC') : '\u25B2'}</span></th>`;
        headerHtml += `<th class="col-pm">PM</th>`;
        headerHtml += `<th class="col-video">V</th>`;
        headerHtml += `<th class="col-ai">AI</th>`;
    });
    headerHtml += '</tr>';
    thead.innerHTML = headerHtml;

    // Filter athletes: exclude DNS for this run
    const allAthletes = cat.athletes.filter(a => {
        const { status } = getRunTime(a);
        return status !== 'DNS';
    });

    // Compute ranking for the active run across ALL athletes (overall rank)
    const ranks = computeRunRanks(allAthletes);

    // Apply team filter AFTER computing overall ranks
    let athletes = allAthletes;
    if (filterTeam) {
        athletes = athletes.filter(a => a.club === filterTeam);
    }

    // Sort
    athletes.sort((a, b) => {
        // Custom sort by column
        if (sortColumn && sortDirection) {
            let va, vb;
            if (sortColumn === 'rank') {
                va = ranks[a.bib] || 9999;
                vb = ranks[b.bib] || 9999;
            } else if (sortColumn === 'time') {
                const ta = getRunTime(a), tb = getRunTime(b);
                va = (ta.status === 'finished' && ta.time != null) ? ta.time : 9999;
                vb = (tb.status === 'finished' && tb.time != null) ? tb.time : 9999;
            } else if (sortColumn === 'bib') {
                va = a.bib;
                vb = b.bib;
            } else if (sortColumn.startsWith('sectionTime_')) {
                const camId = sortColumn.replace('sectionTime_', '');
                va = getSectionTime(a, camId, activeRun) || 9999;
                vb = getSectionTime(b, camId, activeRun) || 9999;
            }
            if (va !== undefined) {
                const cmp = va - vb;
                return sortDirection === 'desc' ? -cmp : cmp;
            }
        }
        // Default: ranked first (by rank), then DNF/DSQ
        const ra = ranks[a.bib], rb = ranks[b.bib];
        if (ra && rb) return ra - rb;
        if (ra) return -1;
        if (rb) return 1;
        return 0;
    });

    let html = athletes.map(a => athleteRow(a, cat, ranks, activeSections)).join('');

    // Append forerunners at the bottom (from Forerunners category, if it exists)
    const frCat = manifest.categories.find(c => c.id === 'Forerunners');
    if (frCat && frCat.athletes) {
        let forerunners = frCat.athletes.filter(a => {
            // Only show forerunners that have montages for any active section
            if (!a.montages) return false;
            return activeSections.some(({ cam }) => {
                const dets = a.montages[cam.id] && a.montages[cam.id][activeRun];
                return Array.isArray(dets) && dets.length > 0;
            });
        });
        // Sort forerunners by section time if a section sort is active
        if (sortColumn && sortColumn.startsWith('sectionTime_') && sortDirection) {
            const camId = sortColumn.replace('sectionTime_', '');
            forerunners.sort((a, b) => {
                const va = getSectionTime(a, camId, activeRun) || 9999;
                const vb = getSectionTime(b, camId, activeRun) || 9999;
                return sortDirection === 'desc' ? vb - va : va - vb;
            });
        }
        if (forerunners.length > 0) {
            html += `<tr class="forerunner-separator"><td colspan="99" class="forerunner-label">Forerunners</td></tr>`;
            html += forerunners.map(a => athleteRow(a, frCat, {}, activeSections, true)).join('');
        }
    }

    tbody.innerHTML = html;
}

function athleteRow(a, cat, ranks, activeSections, isForerunner) {
    const rank = isForerunner ? null : (ranks[a.bib] || null);
    const { time, status } = isForerunner ? { time: null, status: '' } : getRunTime(a);

    let html = `<tr data-bib="${a.bib}" ${isForerunner ? 'class="forerunner-row"' : ''}>`;
    // Official results group — blank for forerunners
    html += `<td class="col-rank"><span class="rank-val">${rank || ''}</span></td>`;
    html += `<td class="col-time">${isForerunner ? '' : formatTimeCell(time, status)}</td>`;
    // Identity group
    html += `<td class="col-bib">${isForerunner ? 'F' : ''}${a.bib}</td>`;
    html += `<td class="col-name">${a.first} ${a.last}</td>`;
    html += `<td class="col-club">${a.club}</td>`;
    // Section times + montage buttons
    activeSections.forEach(({ cam, sectionIdx }) => {
        const montageKey = activeRun;
        const sectionTime = getSectionTime(a, cam.id, montageKey);
        const dets = (a.montages && a.montages[cam.id] && a.montages[cam.id][montageKey]) || [];
        const hasMontage = Array.isArray(dets) && dets.length > 0;
        const detCount = hasMontage ? dets.length : 0;

        // Section time cell (with disclaimer tooltip)
        const stDisclaimer = 'Unofficial estimate from camera analysis, not official timing. For coaching use only.';
        if (sectionTime != null) {
            html += `<td class="col-section-time" title="${stDisclaimer}"><span class="section-time-val">${(Math.floor(sectionTime * 10) / 10).toFixed(1)}</span></td>`;
        } else {
            html += `<td class="col-section-time"><span class="section-time-na">&mdash;</span></td>`;
        }
        // PM button cell — show count if >1 detection
        const pmLabel = hasMontage ? (detCount > 1 ? `PM(${detCount})` : 'PM') : '';
        html += `<td class="col-pm">`;
        html += `<button class="section-btn ${hasMontage ? 'active' : 'inactive'}" ${hasMontage ? `onclick="window._openMontage('${cam.id}',${a.bib},'${cat.id}')"` : ''} title="Photo montage">${pmLabel}</button>`;
        html += `</td>`;
        // Video button cell — show count if >1 video
        const videoCount = hasMontage ? dets.filter(d => d.video).length : 0;
        const hasVideo = videoCount > 0;
        const vLabel = hasVideo ? (videoCount > 1 ? `V(${videoCount})` : 'V') : '';
        html += `<td class="col-video">`;
        html += `<button class="section-btn-video ${hasVideo ? 'active' : 'inactive'}" ${hasVideo ? `onclick="window._openVideo('${cam.id}',${a.bib},'${cat.id}')"` : ''} title="Video clip">${vLabel}</button>`;
        html += `</td>`;
        // AI button cell
        const aiJobKey = `${cam.id}_${a.bib}_${montageKey}`;
        const aiState = window._aiJobStates && window._aiJobStates[aiJobKey];
        let aiLabel = 'AI';
        let aiClass = 'inactive';
        let aiClick = '';
        // Check for pre-computed AI video in manifest first
        const firstDetWithAI = hasMontage ? dets.find(d => d.ai_video) : null;
        if (firstDetWithAI) {
            // Pre-computed AI video available — show checkmark
            aiLabel = 'AI \u2713';
            aiClass = 'complete';
            aiClick = `onclick="window._showPrecomputedAI('${firstDetWithAI.ai_video}',${a.bib})"`;
        } else if (hasVideo) {
            if (aiState && aiState.status === 'running') {
                aiLabel = `${aiState.progress}%`;
                aiClass = 'running';
            } else if (aiState && aiState.status === 'complete') {
                aiLabel = 'AI \u2713';
                aiClass = 'complete';
                aiClick = `onclick="window._showAIResults('${aiJobKey}')"`;
            } else {
                aiClass = 'active';
                const gChar = cat.id.includes('Girls') || cat.id.includes('Women') ? 'g' : 'b';
                const firstDet = dets[0];
                const detIdVal = firstDet ? firstDet.det_id || 'd000' : 'd000';
                aiClick = `onclick="window._runAI('${cam.id}',${a.bib},'${cat.id}','${montageKey}','${gChar}','${detIdVal}')"`;
            }
        }
        html += `<td class="col-ai">`;
        html += `<button class="section-btn-ai ${aiClass}" id="ai-btn-${aiJobKey}" ${aiClick} title="AI Pose Analysis">${aiLabel}</button>`;
        html += `</td>`;
    });
    html += '</tr>';
    return html;
}


// ══════════════════════════════════════════════════════════════════════════
// LIGHTBOX
// ══════════════════════════════════════════════════════════════════════════

let lbList = [];
let lbIdx = 0;
let lbDetIdx = 0;  // Current detection index within athlete's detections array
let lbSelectedFps = null;  // Currently selected FPS (null = default/middle)

window._openMontage = function (camId, bib, catId) {
    const cat = manifest.categories.find(c => c.id === catId);
    if (!cat) return;
    const athlete = cat.athletes.find(a => a.bib === bib);
    if (!athlete || !athlete.montages || !athlete.montages[camId]) return;

    const runKey = activeRun;
    lbList = cat.athletes
        .filter(a => {
            const dets = a.montages && a.montages[camId] && a.montages[camId][runKey];
            return Array.isArray(dets) && dets.length > 0;
        })
        .map(a => ({ athlete: a, camId, catId, runKey }));
    lbIdx = lbList.findIndex(item => item.athlete.bib === bib);
    if (lbIdx < 0) lbIdx = 0;
    lbDetIdx = 0;  // Start at first detection
    showLightbox();
};

let lbCurrentFullUrl = ''; // Track current full-res image URL for download/copy
let vlbCurrentVideoUrl = ''; // Track current video URL for download/copy

function setupLightbox() {
    const lb = document.getElementById('lightbox');
    lb.querySelector('.lb-backdrop').addEventListener('click', closeLB);
    lb.querySelector('.lb-close').addEventListener('click', closeLB);
    lb.querySelector('.lb-prev').addEventListener('click', () => { if (lbIdx > 0) { lbIdx--; lbDetIdx = 0; lbSelectedFps = null; showLightbox(); } });
    lb.querySelector('.lb-next').addEventListener('click', () => { if (lbIdx < lbList.length - 1) { lbIdx++; lbDetIdx = 0; lbSelectedFps = null; showLightbox(); } });

    // Download button - fetch as blob to force download
    document.getElementById('lb-download-btn').addEventListener('click', async () => {
        if (!lbCurrentFullUrl) return;
        const btn = document.getElementById('lb-download-btn');
        const origText = btn.textContent;
        btn.textContent = '⏳ Loading...';
        btn.disabled = true;
        try {
            const response = await fetch(lbCurrentFullUrl);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = lbCurrentFullUrl.split('/').pop();
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('Download failed:', e);
            alert('Download failed. Please try again.');
        }
        btn.textContent = origText;
        btn.disabled = false;
    });

    // Copy link button
    document.getElementById('lb-copy-link-btn').addEventListener('click', () => {
        if (!lbCurrentFullUrl) return;
        const fullUrl = new URL(lbCurrentFullUrl, window.location.href).href;
        navigator.clipboard.writeText(fullUrl).then(() => {
            const btn = document.getElementById('lb-copy-link-btn');
            const orig = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => { btn.textContent = orig; }, 1500);
        });
    });

    document.addEventListener('keydown', e => {
        if (lb.classList.contains('hidden')) return;
        if (e.key === 'Escape') closeLB();
        // UP/DOWN = previous/next athlete
        if (e.key === 'ArrowUp' && lbIdx > 0) { e.preventDefault(); lbIdx--; lbDetIdx = 0; lbSelectedFps = null; showLightbox(); }
        if (e.key === 'ArrowDown' && lbIdx < lbList.length - 1) { e.preventDefault(); lbIdx++; lbDetIdx = 0; lbSelectedFps = null; showLightbox(); }
        // Shift+D toggles delete button
        if (e.key === 'D' && e.shiftKey) {
            const delBtn = document.getElementById('lb-delete-btn');
            if (delBtn) delBtn.style.display = delBtn.style.display === 'none' ? '' : 'none';
        }
    });
}

function showLightbox() {
    const lb = document.getElementById('lightbox');
    const item = lbList[lbIdx];
    if (!item) return;
    const a = item.athlete;
    const runKey = item.runKey || activeRun;
    const dets = a.montages[item.camId][runKey];
    if (!Array.isArray(dets) || dets.length === 0) return;

    // Clamp detection index
    if (lbDetIdx >= dets.length) lbDetIdx = dets.length - 1;
    if (lbDetIdx < 0) lbDetIdx = 0;
    const montage = dets[lbDetIdx];
    if (!montage) return;

    const sectionIdx = manifest.cameras.findIndex(c => c.id === item.camId);
    const runLabel = runKey === 'run1' ? 'Run 1' : 'Run 2';
    const sectionLabel = 'Section ' + (sectionIdx + 1);

    // Determine which image to show (selected FPS or default)
    let thumbUrl = manifest.media_base_url + '/' + montage.thumb;
    let fullUrl = manifest.media_base_url + '/' + montage.full;
    const variants = montage.fps_variants || [];
    if (lbSelectedFps !== null && variants.length > 0) {
        const match = variants.find(v => v.fps === lbSelectedFps);
        if (match) {
            thumbUrl = manifest.media_base_url + '/' + match.thumb;
            fullUrl = manifest.media_base_url + '/' + match.full;
        }
    }
    lbCurrentFullUrl = fullUrl; // Store for download/copy buttons

    lb.classList.remove('hidden');
    document.getElementById('lb-img').src = fullUrl;
    const bibDisplay = a.is_forerunner ? `F${a.bib}` : a.bib;
    document.getElementById('lb-name').textContent = `${a.first} ${a.last} (#${bibDisplay})`;

    // Build details with trigger_time
    let detailText = `${runLabel} | ${sectionLabel}`;
    if (montage.trigger_time) detailText += ` | ${montage.trigger_time}`;
    document.getElementById('lb-details').textContent = detailText;

    lb.querySelector('.lb-prev').style.display = lbIdx > 0 ? '' : 'none';
    lb.querySelector('.lb-next').style.display = lbIdx < lbList.length - 1 ? '' : 'none';

    // Detection navigator (only show if multiple detections)
    const detNav = document.getElementById('lb-det-nav');
    if (detNav) {
        if (dets.length > 1) {
            detNav.style.display = 'flex';
            detNav.innerHTML = `
                <button class="det-nav-btn" id="lb-det-prev" ${lbDetIdx <= 0 ? 'disabled' : ''}>&lsaquo;</button>
                <span class="det-nav-label">${lbDetIdx + 1} / ${dets.length}</span>
                <button class="det-nav-btn" id="lb-det-next" ${lbDetIdx >= dets.length - 1 ? 'disabled' : ''}>&rsaquo;</button>
            `;
            document.getElementById('lb-det-prev').addEventListener('click', () => {
                if (lbDetIdx > 0) { lbDetIdx--; lbSelectedFps = null; showLightbox(); }
            });
            document.getElementById('lb-det-next').addEventListener('click', () => {
                if (lbDetIdx < dets.length - 1) { lbDetIdx++; lbSelectedFps = null; showLightbox(); }
            });
        } else {
            detNav.style.display = 'none';
        }
    }

    // Delete button
    const delBtn = document.getElementById('lb-delete-btn');
    if (delBtn) {
        delBtn.onclick = () => {
            const det = dets[lbDetIdx];
            if (!det) return;
            const gender = a.gender || (item.catId.includes('Girls') || item.catId.includes('Women') ? 'female' : 'male');
            const genderChar = gender === 'female' ? 'g' : 'b';
            window._deleteMontage(manifest.race_slug || '', item.camId, runKey, a.bib, genderChar, det.det_id || 'd000', 'photo');
        };
    }

    // FPS selector buttons
    const fpsSel = document.getElementById('lb-fps-selector');
    if (variants.length > 1) {
        // Determine which fps is currently active
        const activeFps = lbSelectedFps !== null ? lbSelectedFps
            : variants[Math.floor(variants.length / 2)].fps;  // default = middle
        fpsSel.innerHTML = variants.map(v => {
            const isActive = v.fps === activeFps;
            return `<button class="lb-fps-btn${isActive ? ' active' : ''}" data-fps="${v.fps}">${v.fps} img/s</button>`;
        }).join('');
        fpsSel.style.display = 'flex';

        // Click handler for fps buttons
        fpsSel.querySelectorAll('.lb-fps-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                lbSelectedFps = parseFloat(btn.dataset.fps);
                showLightbox();  // Re-render with new fps
            });
        });
    } else {
        fpsSel.style.display = 'none';
    }
}

function closeLB() { document.getElementById('lightbox').classList.add('hidden'); }


// ══════════════════════════════════════════════════════════════════════════
// VIDEO LIGHTBOX
// ══════════════════════════════════════════════════════════════════════════

let vlbList = [];
let vlbIdx = 0;
let vlbDetIdx = 0;  // Current detection index for video lightbox
let vlbSpeed = 1;

window._openVideo = function (camId, bib, catId) {
    const cat = manifest.categories.find(c => c.id === catId);
    if (!cat) return;
    const athlete = cat.athletes.find(a => a.bib === bib);
    if (!athlete || !athlete.montages || !athlete.montages[camId]) return;

    const runKey = activeRun;
    vlbList = cat.athletes
        .filter(a => {
            const dets = a.montages && a.montages[camId] && a.montages[camId][runKey];
            return Array.isArray(dets) && dets.some(d => d.video);
        })
        .map(a => ({ athlete: a, camId, catId, runKey }));
    vlbIdx = vlbList.findIndex(item => item.athlete.bib === bib);
    if (vlbIdx < 0) vlbIdx = 0;
    vlbDetIdx = 0;  // Start at first detection with video
    showVideoLightbox();
};

function setupVideoLightbox() {
    const vlb = document.getElementById('video-lightbox');
    if (!vlb) return;

    vlb.querySelector('.vlb-backdrop').addEventListener('click', closeVLB);
    vlb.querySelector('.vlb-close').addEventListener('click', closeVLB);
    // Prevent clicks on content/controls from bubbling to backdrop and closing the lightbox
    vlb.querySelector('.vlb-content').addEventListener('click', e => e.stopPropagation());
    vlb.querySelector('.vlb-prev').addEventListener('click', () => { if (vlbIdx > 0) { vlbIdx--; vlbDetIdx = 0; showVideoLightbox(); } });
    vlb.querySelector('.vlb-next').addEventListener('click', () => { if (vlbIdx < vlbList.length - 1) { vlbIdx++; vlbDetIdx = 0; showVideoLightbox(); } });

    // Play/Pause button
    const playBtn = document.getElementById('vlb-play-btn');
    const video = document.getElementById('vlb-video');

    // Canvas overlay to show frozen frame when paused (prevents black screen)
    const canvas = document.createElement('canvas');
    canvas.id = 'vlb-canvas-overlay';
    canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;border-radius:var(--radius);pointer-events:none;display:none;';
    // Note: vlb-video-wrap already has position:absolute in CSS which serves as positioning context
    video.parentElement.appendChild(canvas);

    function captureFrameToCanvas() {
        try {
            if (!video.videoWidth || !video.videoHeight) return;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            canvas.style.display = 'block';
        } catch (e) { /* cross-origin or not ready — ignore */ }
    }

    function hideCanvasOverlay() {
        canvas.style.display = 'none';
    }

    function pauseAndShowFrame() {
        video.pause();
        captureFrameToCanvas();
    }

    function togglePlay() {
        if (video.paused) {
            hideCanvasOverlay();
            video.play();
        } else {
            pauseAndShowFrame();
        }
    }
    playBtn.addEventListener('click', togglePlay);
    // Click video itself to toggle play/pause
    video.addEventListener('click', togglePlay);

    video.addEventListener('play', () => { playBtn.innerHTML = '&#10074;&#10074;'; hideCanvasOverlay(); });
    video.addEventListener('pause', () => { playBtn.innerHTML = '&#9654;'; });
    video.addEventListener('ended', () => {
        playBtn.innerHTML = '&#9654;';
        // Loop: restart from beginning
        video.currentTime = 0;
    });

    // Time update → scrubber + time display
    video.addEventListener('timeupdate', () => {
        if (video.duration && !isNaN(video.duration)) {
            const pct = (video.currentTime / video.duration) * 1000;
            document.getElementById('vlb-scrubber').value = pct;
            document.getElementById('vlb-time').textContent =
                `${video.currentTime.toFixed(2)}s / ${video.duration.toFixed(2)}s`;
        }
    });

    // Scrubber seek — pause on scrub
    const scrubber = document.getElementById('vlb-scrubber');
    scrubber.addEventListener('input', () => {
        if (video.duration && !isNaN(video.duration)) {
            video.pause();
            video.currentTime = (scrubber.value / 1000) * video.duration;
        }
    });
    // After scrub seek completes, capture frame so it doesn't go black
    scrubber.addEventListener('change', () => {
        if (video.paused) captureFrameToCanvas();
    });

    // Frame-by-frame stepping (1/30s ≈ 0.033s per frame)
    const FRAME_DURATION = 1 / 30;

    // Update time display after any seek (including frame steps)
    video.addEventListener('seeked', () => {
        if (video.duration && !isNaN(video.duration)) {
            const pct = (video.currentTime / video.duration) * 1000;
            document.getElementById('vlb-scrubber').value = pct;
            document.getElementById('vlb-time').textContent =
                `${video.currentTime.toFixed(3)}s / ${video.duration.toFixed(2)}s`;
        }
    });

    function stepFrame(direction) {
        if (!video.duration || isNaN(video.duration)) return;
        video.pause();
        const newTime = Math.max(0, Math.min(video.currentTime + (direction * FRAME_DURATION), video.duration));
        video.currentTime = newTime;
        // Capture frame to canvas overlay after seek completes
        video.addEventListener('seeked', function onSeeked() {
            video.removeEventListener('seeked', onSeeked);
            captureFrameToCanvas();
        });
    }

    document.getElementById('vlb-frame-back').addEventListener('click', (e) => { e.stopPropagation(); stepFrame(-1); });
    document.getElementById('vlb-frame-fwd').addEventListener('click', (e) => { e.stopPropagation(); stepFrame(1); });

    // Speed buttons
    vlb.querySelectorAll('.vlb-speed').forEach(btn => {
        btn.addEventListener('click', () => {
            vlbSpeed = parseFloat(btn.dataset.speed);
            video.playbackRate = vlbSpeed;
            vlb.querySelectorAll('.vlb-speed').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Download button - fetch as blob to force download
    document.getElementById('vlb-download-btn').addEventListener('click', async () => {
        if (!vlbCurrentVideoUrl) return;
        const btn = document.getElementById('vlb-download-btn');
        const origText = btn.textContent;
        btn.textContent = '⏳ Loading...';
        btn.disabled = true;
        try {
            const response = await fetch(vlbCurrentVideoUrl);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = vlbCurrentVideoUrl.split('/').pop();
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('Download failed:', e);
            alert('Download failed. Please try again.');
        }
        btn.textContent = origText;
        btn.disabled = false;
    });

    // Copy link button
    document.getElementById('vlb-copy-link-btn').addEventListener('click', () => {
        if (!vlbCurrentVideoUrl) return;
        const fullUrl = new URL(vlbCurrentVideoUrl, window.location.href).href;
        navigator.clipboard.writeText(fullUrl).then(() => {
            const btn = document.getElementById('vlb-copy-link-btn');
            const orig = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => { btn.textContent = orig; }, 1500);
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
        if (vlb.classList.contains('hidden')) return;
        if (e.key === 'Escape') closeVLB();
        // UP/DOWN = previous/next athlete
        if (e.key === 'ArrowUp' && vlbIdx > 0) { e.preventDefault(); vlbIdx--; vlbDetIdx = 0; showVideoLightbox(); }
        if (e.key === 'ArrowDown' && vlbIdx < vlbList.length - 1) { e.preventDefault(); vlbIdx++; vlbDetIdx = 0; showVideoLightbox(); }
        // LEFT/RIGHT = frame stepping
        if (e.key === 'ArrowLeft') { e.preventDefault(); stepFrame(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); stepFrame(1); }
        if (e.key === ' ') { e.preventDefault(); togglePlay(); }
        // , and . also step frames
        if (e.key === ',') { e.preventDefault(); stepFrame(-1); }
        if (e.key === '.') { e.preventDefault(); stepFrame(1); }
        // Shift+D toggles delete button
        if (e.key === 'D' && e.shiftKey) {
            const delBtn = document.getElementById('vlb-delete-btn');
            if (delBtn) delBtn.style.display = delBtn.style.display === 'none' ? '' : 'none';
        }
        // 's' cycles through speed settings
        if (e.key === 's' || e.key === 'S') {
            e.preventDefault();
            const speeds = [0.1, 0.25, 0.5, 1];
            const currentIdx = speeds.indexOf(vlbSpeed);
            const nextIdx = (currentIdx + 1) % speeds.length;
            vlbSpeed = speeds[nextIdx];
            video.playbackRate = vlbSpeed;
            vlb.querySelectorAll('.vlb-speed').forEach(b => {
                b.classList.toggle('active', parseFloat(b.dataset.speed) === vlbSpeed);
            });
        }
        // 'c' closes the lightbox
        if (e.key === 'c' || e.key === 'C') {
            e.preventDefault();
            closeVLB();
        }
        // 'g' toggles graph overlay for AI videos
        if (e.key === 'g' || e.key === 'G') {
            const graphControls = document.getElementById('vlb-graph-controls');
            if (graphControls && graphControls.style.display !== 'none' && window._aiGraphOverlay) {
                e.preventDefault();
                const toggleBtn = document.getElementById('vlb-graph-toggle');
                const isVisible = window._aiGraphOverlay.toggle();
                if (toggleBtn) {
                    toggleBtn.textContent = isVisible ? 'Graphs: ON' : 'Graphs: OFF';
                    toggleBtn.classList.toggle('active', isVisible);
                }
            }
        }
    });

    // Graph toggle button for AI videos
    const graphToggleBtn = document.getElementById('vlb-graph-toggle');
    if (graphToggleBtn) {
        graphToggleBtn.addEventListener('click', () => {
            if (window._aiGraphOverlay) {
                const isVisible = window._aiGraphOverlay.toggle();
                graphToggleBtn.textContent = isVisible ? 'Graphs: ON' : 'Graphs: OFF';
                graphToggleBtn.classList.toggle('active', isVisible);
            }
        });
    }

    // Graph opacity slider for AI videos
    const graphOpacitySlider = document.getElementById('vlb-graph-opacity');
    if (graphOpacitySlider) {
        graphOpacitySlider.addEventListener('input', () => {
            if (window._aiGraphOverlay) {
                const val = parseInt(graphOpacitySlider.value) / 100;
                window._aiGraphOverlay.setOpacity(val);
            }
        });
    }

    // Initialize AIGraphOverlay if available
    const graphCanvas = document.getElementById('vlb-graph-canvas');
    console.log('[VLB] Checking AIGraphOverlay:', typeof AIGraphOverlay, 'canvas:', graphCanvas);
    if (graphCanvas && typeof AIGraphOverlay !== 'undefined') {
        window._aiGraphOverlay = new AIGraphOverlay(video, graphCanvas);
        console.log('[VLB] AIGraphOverlay initialized:', window._aiGraphOverlay);
    } else {
        console.warn('[VLB] AIGraphOverlay not available or canvas not found');
    }

    // Make video controls draggable
    const controls = vlb.querySelector('.vlb-controls');
    if (controls) {
        let isDragging = false;
        let startX, startY, origX, origY;

        controls.addEventListener('mousedown', (e) => {
            // Don't drag if clicking on buttons/inputs
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            const rect = controls.getBoundingClientRect();
            origX = rect.left + rect.width / 2;
            origY = rect.top;
            controls.style.transition = 'none';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            controls.style.left = (origX + dx) + 'px';
            controls.style.bottom = 'auto';
            controls.style.top = (origY + dy) + 'px';
            controls.style.transform = 'translateX(-50%)';
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                controls.style.transition = '';
            }
        });
    }
}

function showVideoLightbox() {
    const vlb = document.getElementById('video-lightbox');
    const item = vlbList[vlbIdx];
    if (!item) return;
    const a = item.athlete;
    const runKey = item.runKey || activeRun;
    const dets = a.montages[item.camId][runKey];
    if (!Array.isArray(dets) || dets.length === 0) return;

    // Filter to detections with video
    const videoDets = dets.filter(d => d.video);
    if (videoDets.length === 0) return;
    if (vlbDetIdx >= videoDets.length) vlbDetIdx = videoDets.length - 1;
    if (vlbDetIdx < 0) vlbDetIdx = 0;
    const montage = videoDets[vlbDetIdx];

    const videoUrl = manifest.media_base_url + '/' + montage.video;
    vlbCurrentVideoUrl = videoUrl; // Store for download/copy buttons
    const sectionIdx = manifest.cameras.findIndex(c => c.id === item.camId);
    const runLabel = runKey === 'run1' ? 'Run 1' : 'Run 2';
    const sectionLabel = 'Section ' + (sectionIdx + 1);

    vlb.classList.remove('hidden');

    // Hide canvas overlay from previous video
    const overlay = document.getElementById('vlb-canvas-overlay');
    if (overlay) overlay.style.display = 'none';

    const video = document.getElementById('vlb-video');
    video.src = videoUrl;
    video.playbackRate = vlbSpeed;
    video.load();
    // Show first frame paused, then auto-play
    video.addEventListener('loadeddata', function onLoaded() {
        video.removeEventListener('loadeddata', onLoaded);
        video.currentTime = 0;
        video.play().catch(() => {}); // Autoplay (may fail on mobile without interaction)
    });

    // Build details with trigger_time
    let detailText = `${runLabel} | ${sectionLabel} | Video`;
    if (montage.trigger_time) detailText += ` | ${montage.trigger_time}`;

    const bibDisplayV = a.is_forerunner ? `F${a.bib}` : a.bib;
    document.getElementById('vlb-name').textContent = `${a.first} ${a.last} (#${bibDisplayV})`;
    document.getElementById('vlb-details').textContent = detailText;
    document.getElementById('vlb-time').textContent = '0.00s / 0.00s';
    document.getElementById('vlb-scrubber').value = 0;

    vlb.querySelector('.vlb-prev').style.display = vlbIdx > 0 ? '' : 'none';
    vlb.querySelector('.vlb-next').style.display = vlbIdx < vlbList.length - 1 ? '' : 'none';

    // Detection navigator (only show if multiple video detections)
    const detNav = document.getElementById('vlb-det-nav');
    if (detNav) {
        if (videoDets.length > 1) {
            detNav.style.display = 'flex';
            detNav.innerHTML = `
                <button class="det-nav-btn" id="vlb-det-prev" ${vlbDetIdx <= 0 ? 'disabled' : ''}>&lsaquo;</button>
                <span class="det-nav-label">${vlbDetIdx + 1} / ${videoDets.length}</span>
                <button class="det-nav-btn" id="vlb-det-next" ${vlbDetIdx >= videoDets.length - 1 ? 'disabled' : ''}>&rsaquo;</button>
            `;
            document.getElementById('vlb-det-prev').addEventListener('click', () => {
                if (vlbDetIdx > 0) { vlbDetIdx--; showVideoLightbox(); }
            });
            document.getElementById('vlb-det-next').addEventListener('click', () => {
                if (vlbDetIdx < videoDets.length - 1) { vlbDetIdx++; showVideoLightbox(); }
            });
        } else {
            detNav.style.display = 'none';
        }
    }

    // Delete button
    const delBtn = document.getElementById('vlb-delete-btn');
    if (delBtn) {
        delBtn.onclick = () => {
            const det = videoDets[vlbDetIdx];
            if (!det) return;
            const gender = a.gender || (item.catId.includes('Girls') || item.catId.includes('Women') ? 'female' : 'male');
            const genderChar = gender === 'female' ? 'g' : 'b';
            window._deleteMontage(manifest.race_slug || '', item.camId, runKey, a.bib, genderChar, det.det_id || 'd000', 'video');
        };
    }
}

/**
 * Delete a detection from both the manifest and disk.
 * @param {string} raceSlug
 * @param {string} camId
 * @param {string} runKey
 * @param {number} bib
 * @param {string} genderChar - 'g' or 'b'
 * @param {string} detId - e.g. 'd001'
 * @param {string} source - 'photo' or 'video' (which lightbox initiated)
 */
window._deleteMontage = async function (raceSlug, camId, runKey, bib, genderChar, detId, source) {
    // Derive race_slug from media_base_url if not in manifest
    if (!raceSlug && manifest.media_base_url) {
        raceSlug = manifest.media_base_url.split('/').pop();
    }

    // Prompt for password (cached in sessionStorage for this browser session only)
    let pw = sessionStorage.getItem('skiframes_delete_pw') || '';
    if (!pw) {
        pw = prompt('Enter delete password:');
        if (!pw) return;
    }

    if (!confirm(`Delete detection ${detId} for bib #${bib}?\nThis removes all files (montages + video) for this detection.`)) return;

    try {
        const resp = await fetch('http://localhost:5000/api/montages/delete-detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ race_slug: raceSlug, cam_id: camId, run_key: runKey, bib, gender: genderChar, det_id: detId, password: pw })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            if (resp.status === 403) {
                // Wrong password — clear cached pw so user can re-enter
                sessionStorage.removeItem('skiframes_delete_pw');
            }
            alert('Delete failed: ' + (err.error || resp.statusText));
            return;
        }
        // Password was correct — cache it for this session
        sessionStorage.setItem('skiframes_delete_pw', pw);

        // Update manifest in memory: remove the detection from the array
        for (const cat of manifest.categories) {
            for (const ath of cat.athletes) {
                if (ath.bib !== bib) continue;
                if (!ath.montages || !ath.montages[camId] || !ath.montages[camId][runKey]) continue;
                const dets = ath.montages[camId][runKey];
                if (!Array.isArray(dets)) continue;
                const idx = dets.findIndex(d => d.det_id === detId);
                if (idx >= 0) dets.splice(idx, 1);
                // If array is now empty, clean up
                if (dets.length === 0) {
                    delete ath.montages[camId][runKey];
                    if (Object.keys(ath.montages[camId]).length === 0) delete ath.montages[camId];
                    if (Object.keys(ath.montages).length === 0) delete ath.montages;
                }
            }
        }

        // Re-render results table
        renderResults();

        // Handle lightbox state after deletion
        if (source === 'photo') {
            // Check if athlete still has detections
            const item = lbList[lbIdx];
            if (item) {
                const aDets = item.athlete.montages && item.athlete.montages[camId] && item.athlete.montages[camId][runKey];
                if (!Array.isArray(aDets) || aDets.length === 0) {
                    // No more detections for this athlete — remove from list
                    lbList.splice(lbIdx, 1);
                    if (lbList.length === 0) { closeLB(); return; }
                    if (lbIdx >= lbList.length) lbIdx = lbList.length - 1;
                    lbDetIdx = 0;
                } else if (lbDetIdx >= aDets.length) {
                    lbDetIdx = aDets.length - 1;
                }
            }
            lbSelectedFps = null;
            showLightbox();
        } else {
            // Video lightbox
            const item = vlbList[vlbIdx];
            if (item) {
                const aDets = item.athlete.montages && item.athlete.montages[camId] && item.athlete.montages[camId][runKey];
                const videoDets = Array.isArray(aDets) ? aDets.filter(d => d.video) : [];
                if (videoDets.length === 0) {
                    vlbList.splice(vlbIdx, 1);
                    if (vlbList.length === 0) { closeVLB(); return; }
                    if (vlbIdx >= vlbList.length) vlbIdx = vlbList.length - 1;
                    vlbDetIdx = 0;
                } else if (vlbDetIdx >= videoDets.length) {
                    vlbDetIdx = videoDets.length - 1;
                }
            }
            showVideoLightbox();
        }
    } catch (e) {
        alert('Delete error: ' + e.message);
    }
};

function closeVLB() {
    const vlb = document.getElementById('video-lightbox');
    const video = document.getElementById('vlb-video');
    video.pause();
    video.removeAttribute('src');
    video.load(); // Release video resources
    // Hide canvas overlay
    const overlay = document.getElementById('vlb-canvas-overlay');
    if (overlay) overlay.style.display = 'none';
    // Hide and reset graph controls
    const graphControls = document.getElementById('vlb-graph-controls');
    const graphHint = document.getElementById('vlb-graph-hint');
    const graphToggleBtn = document.getElementById('vlb-graph-toggle');
    if (graphControls) graphControls.style.display = 'none';
    if (graphHint) graphHint.style.display = 'none';
    if (graphToggleBtn) {
        graphToggleBtn.textContent = 'Graphs: OFF';
        graphToggleBtn.classList.remove('active');
    }
    // Hide graph overlay
    if (window._aiGraphOverlay) {
        window._aiGraphOverlay.setVisible(false);
    }
    video.style.filter = '';
    vlb.classList.add('hidden');
}


// ══════════════════════════════════════════════════════════════════════════
// AI POSE ANALYSIS
// ══════════════════════════════════════════════════════════════════════════

window._aiJobStates = {};  // aiJobKey -> {job_id, status, progress, results}

window._runAI = function (camId, bib, catId, runKey, genderChar, detId) {
    const aiJobKey = `${camId}_${bib}_${runKey}`;

    // Derive race_slug
    let raceSlug = manifest.race_slug || '';
    if (!raceSlug && manifest.media_base_url) {
        raceSlug = manifest.media_base_url.split('/').pop();
    }

    // Set running state immediately
    window._aiJobStates[aiJobKey] = { status: 'running', progress: 0, results: null };
    _updateAIButton(aiJobKey);

    // POST to start analysis
    fetch('http://localhost:5000/api/ai/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ race_slug: raceSlug, cam_id: camId, run_key: runKey, bib, gender: genderChar, det_id: detId })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            window._aiJobStates[aiJobKey] = { status: 'error', progress: 0, results: null };
            _updateAIButton(aiJobKey);
            alert('AI analysis failed: ' + data.error);
            return;
        }
        window._aiJobStates[aiJobKey].job_id = data.job_id;
        // Start polling
        _pollAIJob(aiJobKey, data.job_id);
    })
    .catch(e => {
        window._aiJobStates[aiJobKey] = { status: 'error', progress: 0, results: null };
        _updateAIButton(aiJobKey);
        alert('AI request failed: ' + e.message);
    });
};

function _pollAIJob(aiJobKey, jobId) {
    fetch(`http://localhost:5000/api/ai/status/${jobId}`)
        .then(r => r.json())
        .then(data => {
            const state = window._aiJobStates[aiJobKey];
            if (!state) return;

            state.progress = data.progress || 0;
            state.status_msg = data.status_msg || '';

            if (data.status === 'complete') {
                state.status = 'complete';
                state.results = data.results;
                _updateAIButton(aiJobKey);
                // Auto-open the annotated video
                window._showAIResults(aiJobKey);
                return;
            }
            if (data.status === 'error') {
                state.status = 'error';
                state.errorMsg = data.error || 'Unknown error';
                _updateAIButton(aiJobKey);
                alert('AI analysis error: ' + state.errorMsg);
                return;
            }

            // Still running — update and poll again (faster during loading phase)
            _updateAIButton(aiJobKey);
            const pollInterval = state.progress === 0 ? 1000 : 2000;
            setTimeout(() => _pollAIJob(aiJobKey, jobId), pollInterval);
        })
        .catch(() => {
            setTimeout(() => _pollAIJob(aiJobKey, jobId), 3000);
        });
}

function _updateAIButton(aiJobKey) {
    const btn = document.getElementById(`ai-btn-${aiJobKey}`);
    if (!btn) return;

    const state = window._aiJobStates[aiJobKey];
    if (!state) return;

    btn.className = 'section-btn-ai';
    if (state.status === 'running') {
        if (state.progress === 0 && state.status_msg) {
            // Show short status while model loads (truncate for small button)
            const msg = state.status_msg.replace('Loading AI model...', 'Loading...').replace('Initializing pose analyzer...', 'Init...').replace('Encoding video for browser...', 'Encoding...');
            btn.textContent = msg;
        } else {
            btn.textContent = `${state.progress}%`;
        }
        btn.classList.add('running');
        btn.onclick = null;
    } else if (state.status === 'complete') {
        btn.textContent = 'AI \u2713';
        btn.classList.add('complete');
        btn.onclick = () => window._showAIResults(aiJobKey);
    } else if (state.status === 'error') {
        btn.textContent = 'AI \u2717';
        btn.classList.add('error');
        btn.onclick = null;
        setTimeout(() => {
            // Reset after 3s
            delete window._aiJobStates[aiJobKey];
            renderResults();
        }, 3000);
    }
}

window._showAIResults = function (aiJobKey) {
    const state = window._aiJobStates[aiJobKey];
    if (!state || !state.results || !state.results.ai_video) return;

    // Open the AI-annotated video in the video lightbox
    const aiVideoUrl = manifest.media_base_url + '/' + state.results.ai_video;
    vlbCurrentVideoUrl = aiVideoUrl; // Store for download/copy buttons

    // Parse aiJobKey to get athlete info: "Cam1_10_run1"
    const parts = aiJobKey.split('_');
    const camId = parts[0];
    const bib = parseInt(parts[1]);
    const runKey = parts.slice(2).join('_');

    // Find the athlete
    let athlete = null;
    for (const cat of manifest.categories || []) {
        for (const a of cat.athletes || []) {
            if (a.bib === bib) { athlete = a; break; }
        }
        if (athlete) break;
    }

    const vlb = document.getElementById('video-lightbox');
    vlb.classList.remove('hidden');

    // Hide canvas overlay from previous video
    const overlay = document.getElementById('vlb-canvas-overlay');
    if (overlay) overlay.style.display = 'none';

    const video = document.getElementById('vlb-video');
    video.src = aiVideoUrl;
    video.playbackRate = vlbSpeed;
    video.load();
    video.addEventListener('loadeddata', function onLoaded() {
        video.removeEventListener('loadeddata', onLoaded);
        video.currentTime = 0;
        video.play().catch(() => {});
    });

    const bibStr1 = athlete?.is_forerunner ? `F${athlete.bib}` : (athlete?.bib || bib);
    const nameStr = athlete ? `${athlete.first} ${athlete.last} (#${bibStr1})` : `Bib #${bib}`;
    document.getElementById('vlb-name').textContent = nameStr;
    document.getElementById('vlb-details').textContent = `AI Pose Analysis | ${state.results.frames_analyzed} poses / ${state.results.total_frames} frames`;
    document.getElementById('vlb-time').textContent = '0.00s / 0.00s';
    document.getElementById('vlb-scrubber').value = 0;

    // Hide navigation arrows (single AI video, not part of athlete list)
    vlb.querySelector('.vlb-prev').style.display = 'none';
    vlb.querySelector('.vlb-next').style.display = 'none';

    // Hide detection nav and delete button for AI view
    const detNav = document.getElementById('vlb-det-nav');
    if (detNav) detNav.style.display = 'none';
    const delBtn = document.getElementById('vlb-delete-btn');
    if (delBtn) delBtn.style.display = 'none';

    // Show graph controls for AI videos
    const graphControls = document.getElementById('vlb-graph-controls');
    const graphHint = document.getElementById('vlb-graph-hint');
    const graphToggleBtn = document.getElementById('vlb-graph-toggle');
    if (graphControls) graphControls.style.display = 'flex';
    if (graphHint) graphHint.style.display = '';

    // Reset graph toggle state (graphs off by default)
    if (graphToggleBtn) {
        graphToggleBtn.textContent = 'Graphs: OFF';
        graphToggleBtn.classList.remove('active');
    }

    // Load metrics JSON for interactive graph overlay
    if (window._aiGraphOverlay) {
        window._aiGraphOverlay.setVisible(false);  // Start with graphs hidden
        // Metrics JSON has same path as video but with .json extension
        const metricsJsonUrl = aiVideoUrl.replace(/_ai\.mp4$/, '_ai.json').replace(/\.mp4$/, '.json');
        console.log('[AI] Loading metrics from:', metricsJsonUrl);
        // Wait for video to have dimensions before loading metrics
        const loadMetricsWhenReady = () => {
            window._aiGraphOverlay.loadMetrics(metricsJsonUrl).then(loaded => {
                if (loaded) {
                    console.log('[AI] Metrics loaded, triggering resize');
                    window._aiGraphOverlay._onResize();
                }
            });
        };
        if (video.videoWidth > 0) {
            loadMetricsWhenReady();
        } else {
            video.addEventListener('loadedmetadata', function onMeta() {
                video.removeEventListener('loadedmetadata', onMeta);
                loadMetricsWhenReady();
            });
        }
    }
};

/**
 * Show a pre-computed AI analysis video from the manifest.
 * Opens the video lightbox directly with the AI video URL.
 */
window._showPrecomputedAI = function(aiVideoRelPath, bib) {
    const aiVideoUrl = manifest.media_base_url + '/' + aiVideoRelPath;
    vlbCurrentVideoUrl = aiVideoUrl; // Store for download/copy buttons

    // Find the athlete by bib
    let athlete = null;
    for (const cat of manifest.categories || []) {
        for (const a of cat.athletes || []) {
            if (a.bib === bib) { athlete = a; break; }
        }
        if (athlete) break;
    }

    const vlb = document.getElementById('video-lightbox');
    vlb.classList.remove('hidden');

    // Hide canvas overlay from previous video
    const overlay = document.getElementById('vlb-canvas-overlay');
    if (overlay) overlay.style.display = 'none';

    const video = document.getElementById('vlb-video');
    video.src = aiVideoUrl;
    video.playbackRate = vlbSpeed;
    video.load();
    video.addEventListener('loadeddata', function onLoaded() {
        video.removeEventListener('loadeddata', onLoaded);
        video.currentTime = 0;
        video.play().catch(() => {});
    });

    const bibStr2 = athlete?.is_forerunner ? `F${athlete.bib}` : (athlete?.bib || bib);
    const nameStr = athlete ? `${athlete.first} ${athlete.last} (#${bibStr2})` : `Bib #${bib}`;
    document.getElementById('vlb-name').textContent = nameStr;

    // Show AI version from manifest
    const ver = manifest.ai_version || 'pre-computed';
    document.getElementById('vlb-details').textContent = `AI Pose Analysis | ${ver}`;
    document.getElementById('vlb-time').textContent = '0.00s / 0.00s';
    document.getElementById('vlb-scrubber').value = 0;

    // Hide navigation arrows (single AI video, not part of athlete list)
    vlb.querySelector('.vlb-prev').style.display = 'none';
    vlb.querySelector('.vlb-next').style.display = 'none';

    // Hide detection nav and delete button for AI view
    const detNav = document.getElementById('vlb-det-nav');
    if (detNav) detNav.style.display = 'none';
    const delBtn = document.getElementById('vlb-delete-btn');
    if (delBtn) delBtn.style.display = 'none';

    // Show graph controls for AI videos
    const graphControls = document.getElementById('vlb-graph-controls');
    const graphHint = document.getElementById('vlb-graph-hint');
    const graphToggleBtn = document.getElementById('vlb-graph-toggle');
    if (graphControls) graphControls.style.display = 'flex';
    if (graphHint) graphHint.style.display = '';

    // Reset graph toggle state (graphs off by default)
    if (graphToggleBtn) {
        graphToggleBtn.textContent = 'Graphs: OFF';
        graphToggleBtn.classList.remove('active');
    }

    // Load metrics JSON for interactive graph overlay
    if (window._aiGraphOverlay) {
        window._aiGraphOverlay.setVisible(false);  // Start with graphs hidden
        // Metrics JSON has same path as video but with .json extension
        const metricsJsonUrl = aiVideoUrl.replace(/_ai\.mp4$/, '_ai.json').replace(/\.mp4$/, '.json');
        console.log('[AI] Loading metrics from:', metricsJsonUrl);
        // Wait for video to have dimensions before loading metrics
        const loadMetricsWhenReady = () => {
            window._aiGraphOverlay.loadMetrics(metricsJsonUrl).then(loaded => {
                if (loaded) {
                    console.log('[AI] Metrics loaded, triggering resize');
                    window._aiGraphOverlay._onResize();
                }
            });
        };
        if (video.videoWidth > 0) {
            loadMetricsWhenReady();
        } else {
            video.addEventListener('loadedmetadata', function onMeta() {
                video.removeEventListener('loadedmetadata', onMeta);
                loadMetricsWhenReady();
            });
        }
    }
};


// ── Virtual Race ─────────────────────────────────────────────────────────
/**
 * Virtual Race - Side-by-side athlete comparison
 * Modes: PM (Photo Montage), V (Video)
 */
const VirtualRace = {
    modal: null,
    mode: 'V',
    layout: 'side',      // 'side', 'stack', or 'ghost'
    athletes: [],        // Athletes with video data (filtered by run/section)
    allAthleteData: [],  // Raw athlete data from manifest
    athleteA: null,
    athleteB: null,
    videoA: null,
    videoB: null,
    isPlaying: false,
    playbackRate: 1,
    syncRAF: null,
    selectedRun: 'run1',
    selectedCam: null,
    availableRuns: [],           // e.g. ['run1', 'run2']
    sectionsPerRun: {},          // e.g. { run1: ['Cam1','Cam3'], run2: ['Cam2'] }

    open() {
        this.modal = document.getElementById('virtualRaceModal');
        if (!this.modal || !manifest) return;

        // Gather ALL athlete data from manifest (run/section agnostic)
        this.allAthleteData = [];
        const seen = new Set();
        manifest.categories.forEach(cat => {
            const gender = cat.id.includes('Girls') || cat.id.includes('Women') ? 'g' : 'b';
            cat.athletes.forEach(a => {
                const key = `${gender}${a.bib}`;
                if (!seen.has(key) && a.montages) {
                    seen.add(key);
                    this.allAthleteData.push({
                        bib: a.bib,
                        name: `${a.first} ${a.last}`,
                        team: a.club || '',
                        gender,
                        montages: a.montages
                    });
                }
            });
        });
        this.allAthleteData.sort((a, b) => a.bib - b.bib);

        // Discover available runs and sections (cameras) with valid video + section_time
        const runCamSet = {};  // { run1: Set('Cam1','Cam3'), run2: Set('Cam2') }
        this.allAthleteData.forEach(a => {
            for (const camId of Object.keys(a.montages)) {
                for (const runKey of Object.keys(a.montages[camId])) {
                    const dets = a.montages[camId][runKey];
                    if (Array.isArray(dets) && dets.some(d => d.video && d.section_time)) {
                        if (!runCamSet[runKey]) runCamSet[runKey] = new Set();
                        runCamSet[runKey].add(camId);
                    }
                }
            }
        });
        this.availableRuns = Object.keys(runCamSet).sort();
        this.sectionsPerRun = {};
        for (const [runKey, camSet] of Object.entries(runCamSet)) {
            // Sort cameras by index so Section 1 < Section 2 < Section 3
            const camOrder = (manifest.cameras || []).map(c => c.id);
            this.sectionsPerRun[runKey] = [...camSet].sort((a, b) => camOrder.indexOf(a) - camOrder.indexOf(b));
        }

        // Default selection
        if (!this.availableRuns.includes(this.selectedRun)) {
            this.selectedRun = this.availableRuns[0] || 'run1';
        }
        const sections = this.sectionsPerRun[this.selectedRun] || [];
        if (!sections.includes(this.selectedCam)) {
            this.selectedCam = sections[0] || null;
        }

        // Render run/section toggle buttons
        this._renderRunTabs();
        this._renderSectionTabs();

        // Filter athletes for selected run/section
        this._refreshAthletes();

        // Update mode tabs
        const hasVideos = this.athletes.length >= 2;
        this._setTabEnabled('V', hasVideos);
        this._setTabEnabled('PM', true);
        this._setTabEnabled('AI', true);
        this.mode = hasVideos ? 'V' : 'PM';
        this._setActiveTab(this.mode);

        // Setup layout tabs
        this.modal.querySelectorAll('.vr-layout-tab').forEach(tab => {
            tab.addEventListener('click', () => this.setLayout(tab.dataset.layout));
        });
        this._setActiveLayout(this.layout);

        // Setup ghost opacity slider
        const ghostSlider = document.getElementById('vrGhostOpacity');
        if (ghostSlider) {
            ghostSlider.addEventListener('input', (e) => this.setGhostOpacity(e.target.value));
        }

        // Show modal
        this.modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        this.renderContent();

        // Keyboard
        this._keyHandler = (e) => this._onKeydown(e);
        document.addEventListener('keydown', this._keyHandler);
    },

    setLayout(layout) {
        this.layout = layout;
        this._setActiveLayout(layout);
        // Show/hide ghost opacity slider
        const slider = document.getElementById('vrGhostSlider');
        if (slider) slider.style.display = layout === 'ghost' ? 'flex' : 'none';
        this.renderContent();
    },

    _setActiveLayout(layout) {
        this.modal?.querySelectorAll('.vr-layout-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.layout === layout);
        });
    },

    setGhostOpacity(value) {
        const container = document.getElementById('vrContent');
        if (container) {
            container.style.setProperty('--ghost-opacity', value / 100);
        }
        const label = document.getElementById('vrGhostValue');
        if (label) label.textContent = value + '%';
    },

    close() {
        if (!this.modal) return;
        this._stopSync();
        this._pauseVideos();
        this.modal.classList.add('hidden');
        document.body.style.overflow = '';
        this.videoA = null;
        this.videoB = null;
        this.isPlaying = false;
        if (this._keyHandler) {
            document.removeEventListener('keydown', this._keyHandler);
            this._keyHandler = null;
        }
    },

    setMode(mode) {
        const tab = this.modal.querySelector(`.vr-mode-tab[data-mode="${mode}"]`);
        if (tab && tab.classList.contains('disabled')) return;
        this._pauseVideos();
        this._stopSync();
        this.mode = mode;
        this._setActiveTab(mode);
        this.isPlaying = false;
        this.renderContent();
    },

    selectAthlete(side, key) {
        const athlete = this.athletes.find(a => `${a.gender}${a.bib}` === key);
        if (!athlete) return;
        if (side === 'A') this.athleteA = athlete;
        else this.athleteB = athlete;
        this._pauseVideos();
        this._stopSync();
        this.isPlaying = false;
        this.renderContent();
    },

    // ── Run / Section toggle methods ──────────────────────────────────────

    setRun(runKey) {
        if (runKey === this.selectedRun) return;
        this.selectedRun = runKey;
        this._renderRunTabs();
        // Update section to first available for this run
        const sections = this.sectionsPerRun[runKey] || [];
        if (!sections.includes(this.selectedCam)) {
            this.selectedCam = sections[0] || null;
        }
        this._renderSectionTabs();
        this._pauseVideos();
        this._stopSync();
        this.isPlaying = false;
        this._refreshAthletes();
        this.renderContent();
    },

    setSection(camId) {
        if (camId === this.selectedCam) return;
        this.selectedCam = camId;
        this._renderSectionTabs();
        this._pauseVideos();
        this._stopSync();
        this.isPlaying = false;
        this._refreshAthletes();
        this.renderContent();
    },

    _renderRunTabs() {
        const container = document.getElementById('vrRunTabs');
        if (!container) return;
        if (this.availableRuns.length <= 1) {
            // Only one run — still show it but no toggling needed
        }
        container.innerHTML = this.availableRuns.map(runKey => {
            const num = runKey.replace('run', '');
            const active = runKey === this.selectedRun ? ' active' : '';
            return `<button class="vr-run-tab${active}" data-run="${runKey}" onclick="VirtualRace.setRun('${runKey}')">Run ${num}</button>`;
        }).join('');
    },

    _renderSectionTabs() {
        const container = document.getElementById('vrSectionTabs');
        if (!container) return;
        const sections = this.sectionsPerRun[this.selectedRun] || [];
        const camOrder = (manifest.cameras || []).map(c => c.id);
        container.innerHTML = sections.map(camId => {
            const idx = camOrder.indexOf(camId) + 1;
            const active = camId === this.selectedCam ? ' active' : '';
            return `<button class="vr-section-tab${active}" data-cam="${camId}" onclick="VirtualRace.setSection('${camId}')">Section ${idx}</button>`;
        }).join('');
    },

    _refreshAthletes() {
        // Filter allAthleteData to athletes with video + section_time for selected run/cam
        const prevA = this.athleteA ? `${this.athleteA.gender}${this.athleteA.bib}` : null;
        const prevB = this.athleteB ? `${this.athleteB.gender}${this.athleteB.bib}` : null;

        this.athletes = [];
        if (!this.selectedCam || !this.selectedRun) {
            this._populateDropdowns();
            return;
        }

        this.allAthleteData.forEach(a => {
            const camData = a.montages[this.selectedCam];
            if (!camData) return;
            const runData = camData[this.selectedRun];
            if (!Array.isArray(runData)) return;
            const det = runData.find(d => d.video && d.section_time);
            if (!det) return;

            this.athletes.push({
                bib: a.bib,
                name: a.name,
                team: a.team,
                gender: a.gender,
                videoUrl: manifest.media_base_url + '/' + det.video,
                aiVideoUrl: det.ai_video ? manifest.media_base_url + '/' + det.ai_video : null,
                duration: det.section_time,
                thumb: det.thumb ? manifest.media_base_url + '/' + det.thumb : null,
                full: det.full ? manifest.media_base_url + '/' + det.full : null,
                fpsVariants: det.fps_variants || []
            });
        });

        this._populateDropdowns();

        // Try to re-select previous athletes, otherwise pick first two
        const findA = this.athletes.find(a => `${a.gender}${a.bib}` === prevA);
        const findB = this.athletes.find(a => `${a.gender}${a.bib}` === prevB);

        if (findA) {
            this.athleteA = findA;
            document.getElementById('vrSelectA').value = prevA;
        } else if (this.athletes.length >= 1) {
            this.athleteA = this.athletes[0];
            document.getElementById('vrSelectA').value = `${this.athleteA.gender}${this.athleteA.bib}`;
        } else {
            this.athleteA = null;
        }

        if (findB && findB !== this.athleteA) {
            this.athleteB = findB;
            document.getElementById('vrSelectB').value = prevB;
        } else if (this.athletes.length >= 2) {
            this.athleteB = this.athletes[1];
            document.getElementById('vrSelectB').value = `${this.athleteB.gender}${this.athleteB.bib}`;
        } else {
            this.athleteB = null;
        }

        // Update mode tabs
        const hasVideos = this.athletes.length >= 2;
        this._setTabEnabled('V', hasVideos);
    },

    renderContent() {
        const container = document.getElementById('vrContent');
        if (!container) return;

        if (this.mode === 'V') {
            this._renderVideoMode(container);
        } else if (this.mode === 'AI') {
            this._renderAIMode(container);
        } else {
            this._renderPMMode(container);
        }
        this._updateDiff();
    },

    _renderVideoMode(container) {
        if (!this.athleteA || !this.athleteB) {
            container.innerHTML = '<div class="vr-empty">Select two athletes to compare</div>';
            document.getElementById('vrControlsSection').classList.add('hidden');
            return;
        }

        const layoutClass = this.layout === 'stack' ? 'layout-stack' : (this.layout === 'ghost' ? 'layout-ghost' : '');
        container.innerHTML = `
            <div class="vr-panels ${layoutClass}">
                <div class="vr-panel">
                    <div class="vr-panel-label">#${this.athleteA.bib} ${this.athleteA.name}</div>
                    <div class="vr-video-wrap">
                        <video id="vrVideoA" preload="auto" playsinline muted>
                            <source src="${this.athleteA.videoUrl}" type="video/mp4">
                        </video>
                    </div>
                    <div class="vr-panel-info">
                        ${this.athleteA.duration ? this.athleteA.duration.toFixed(2) + 's' : '-'}
                        <span class="vr-panel-meta">${this.athleteA.team}</span>
                    </div>
                </div>
                <div class="vr-panel">
                    <div class="vr-panel-label">#${this.athleteB.bib} ${this.athleteB.name}</div>
                    <div class="vr-video-wrap">
                        <video id="vrVideoB" preload="auto" playsinline muted>
                            <source src="${this.athleteB.videoUrl}" type="video/mp4">
                        </video>
                    </div>
                    <div class="vr-panel-info">
                        ${this.athleteB.duration ? this.athleteB.duration.toFixed(2) + 's' : '-'}
                        <span class="vr-panel-meta">${this.athleteB.team}</span>
                    </div>
                </div>
            </div>
        `;

        this.videoA = document.getElementById('vrVideoA');
        this.videoB = document.getElementById('vrVideoB');
        if (this.videoA) this.videoA.playbackRate = this.playbackRate;
        if (this.videoB) this.videoB.playbackRate = this.playbackRate;
        document.getElementById('vrControlsSection').classList.remove('hidden');
    },

    _renderPMMode(container) {
        // For PM mode, show full-res images side by side
        if (!this.athleteA || !this.athleteB) {
            container.innerHTML = '<div class="vr-empty">Select two athletes to compare</div>';
            document.getElementById('vrControlsSection').classList.add('hidden');
            return;
        }

        // Use montage from selected run/section stored on athlete object
        const imgA = this.athleteA.full || null;
        const imgB = this.athleteB.full || null;

        const layoutClass = this.layout === 'stack' ? 'layout-stack' : (this.layout === 'ghost' ? 'layout-ghost' : '');
        container.innerHTML = `
            <div class="vr-panels ${layoutClass}">
                <div class="vr-panel">
                    <div class="vr-panel-label">#${this.athleteA.bib} ${this.athleteA.name}</div>
                    <div class="vr-montage-wrap">
                        ${imgA ? `<img src="${imgA}" alt="Montage A">` : '<div class="vr-empty">No montage</div>'}
                    </div>
                </div>
                <div class="vr-panel">
                    <div class="vr-panel-label">#${this.athleteB.bib} ${this.athleteB.name}</div>
                    <div class="vr-montage-wrap">
                        ${imgB ? `<img src="${imgB}" alt="Montage B">` : '<div class="vr-empty">No montage</div>'}
                    </div>
                </div>
            </div>
        `;
        document.getElementById('vrControlsSection').classList.add('hidden');
    },

    _renderAIMode(container) {
        // AI mode: show AI-analyzed videos side by side
        if (!this.athleteA || !this.athleteB) {
            container.innerHTML = '<div class="vr-empty">Select two athletes to compare</div>';
            document.getElementById('vrControlsSection').classList.add('hidden');
            return;
        }

        // Use AI video from selected run/section stored on athlete object
        const aiA = this.athleteA.aiVideoUrl || null;
        const aiB = this.athleteB.aiVideoUrl || null;

        if (!aiA && !aiB) {
            container.innerHTML = '<div class="vr-empty">No AI videos available. Run AI analysis first.</div>';
            document.getElementById('vrControlsSection').classList.add('hidden');
            return;
        }

        const layoutClass = this.layout === 'stack' ? 'layout-stack' : (this.layout === 'ghost' ? 'layout-ghost' : '');
        container.innerHTML = `
            <div class="vr-panels ${layoutClass}">
                <div class="vr-panel">
                    <div class="vr-panel-label">#${this.athleteA.bib} ${this.athleteA.name}</div>
                    <div class="vr-video-wrap">
                        ${aiA ? `<video id="vrVideoA" preload="auto" playsinline muted><source src="${aiA}" type="video/mp4"></video>` : '<div class="vr-empty">No AI video</div>'}
                    </div>
                    <div class="vr-panel-info">
                        ${this.athleteA.duration ? this.athleteA.duration.toFixed(2) + 's' : '-'}
                        <span class="vr-panel-meta">${this.athleteA.team}</span>
                    </div>
                </div>
                <div class="vr-panel">
                    <div class="vr-panel-label">#${this.athleteB.bib} ${this.athleteB.name}</div>
                    <div class="vr-video-wrap">
                        ${aiB ? `<video id="vrVideoB" preload="auto" playsinline muted><source src="${aiB}" type="video/mp4"></video>` : '<div class="vr-empty">No AI video</div>'}
                    </div>
                    <div class="vr-panel-info">
                        ${this.athleteB.duration ? this.athleteB.duration.toFixed(2) + 's' : '-'}
                        <span class="vr-panel-meta">${this.athleteB.team}</span>
                    </div>
                </div>
            </div>
        `;

        this.videoA = document.getElementById('vrVideoA');
        this.videoB = document.getElementById('vrVideoB');
        if (this.videoA) this.videoA.playbackRate = this.playbackRate;
        if (this.videoB) this.videoB.playbackRate = this.playbackRate;

        document.getElementById('vrControlsSection').classList.remove('hidden');
    },

    togglePlay() {
        if (this.isPlaying) this._pauseVideos();
        else this._playVideos();
    },

    _playVideos() {
        if (!this.videoA || !this.videoB) return;
        Promise.all([this.videoA.play(), this.videoB.play()]).then(() => {
            this.isPlaying = true;
            this._updatePlayBtn();
            this._startSync();
        }).catch(e => console.log('Autoplay prevented:', e));
    },

    _pauseVideos() {
        if (this.videoA && !this.videoA.paused) this.videoA.pause();
        if (this.videoB && !this.videoB.paused) this.videoB.pause();
        this.isPlaying = false;
        this._updatePlayBtn();
        this._stopSync();
    },

    _startSync() {
        this._stopSync();
        const sync = () => {
            if (!this.isPlaying || !this.videoA || !this.videoB) return;
            const drift = Math.abs(this.videoA.currentTime - this.videoB.currentTime);
            if (drift > 0.1) this.videoB.currentTime = this.videoA.currentTime;
            this._updateTimeDisplay();
            this.syncRAF = requestAnimationFrame(sync);
        };
        this.syncRAF = requestAnimationFrame(sync);
    },

    _stopSync() {
        if (this.syncRAF) {
            cancelAnimationFrame(this.syncRAF);
            this.syncRAF = null;
        }
    },

    setSpeed(rate) {
        this.playbackRate = rate;
        if (this.videoA) this.videoA.playbackRate = rate;
        if (this.videoB) this.videoB.playbackRate = rate;
        this.modal.querySelectorAll('.vr-speed-btn').forEach(btn => {
            btn.classList.toggle('active', parseFloat(btn.dataset.speed) === rate);
        });
    },

    frameStep(delta) {
        this._pauseVideos();
        const step = delta / 30;
        if (this.videoA) this.videoA.currentTime = Math.max(0, this.videoA.currentTime + step);
        if (this.videoB) this.videoB.currentTime = Math.max(0, this.videoB.currentTime + step);
        this._updateTimeDisplay();
    },

    restart() {
        if (this.videoA) this.videoA.currentTime = 0;
        if (this.videoB) this.videoB.currentTime = 0;
        this._updateTimeDisplay();
    },

    _populateDropdowns() {
        const selA = document.getElementById('vrSelectA');
        const selB = document.getElementById('vrSelectB');
        if (!selA || !selB) return;
        const options = this.athletes.map(a => {
            const time = a.duration ? ` (${a.duration.toFixed(2)}s)` : '';
            return `<option value="${a.gender}${a.bib}">#${a.bib} ${a.name}${time}</option>`;
        }).join('');
        selA.innerHTML = options;
        selB.innerHTML = options;
    },

    _setTabEnabled(mode, enabled) {
        const tab = this.modal?.querySelector(`.vr-mode-tab[data-mode="${mode}"]`);
        if (tab) {
            tab.classList.toggle('disabled', !enabled);
        }
    },

    _setActiveTab(mode) {
        this.modal?.querySelectorAll('.vr-mode-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.mode === mode);
        });
    },

    _updatePlayBtn() {
        const btn = document.getElementById('vrPlayBtn');
        if (btn) btn.textContent = this.isPlaying ? '⏸ Pause' : '▶ Play';
    },

    _updateTimeDisplay() {
        const display = document.getElementById('vrTimeDisplay');
        if (display && this.videoA) display.textContent = `${this.videoA.currentTime.toFixed(2)}s`;
    },

    _updateDiff() {
        const diffEl = document.getElementById('vrDiff');
        if (!diffEl) return;
        if (this.athleteA?.duration && this.athleteB?.duration) {
            const diff = this.athleteB.duration - this.athleteA.duration;
            const sign = diff > 0 ? '+' : '';
            diffEl.textContent = `${sign}${diff.toFixed(2)}s`;
            diffEl.className = 'vr-diff ' + (diff > 0 ? 'vr-diff-behind' : diff < 0 ? 'vr-diff-ahead' : '');
        } else {
            diffEl.textContent = '';
            diffEl.className = 'vr-diff';
        }
    },

    _onKeydown(e) {
        if (!this.modal || this.modal.classList.contains('hidden')) return;
        switch (e.key) {
            case 'Escape': this.close(); break;
            case ' ': e.preventDefault(); this.togglePlay(); break;
            case 'ArrowLeft': e.preventDefault(); this.frameStep(-1); break;
            case 'ArrowRight': e.preventDefault(); this.frameStep(1); break;
            case 'r': case 'R': e.preventDefault(); this.restart(); break;
        }
    }
};

// Make VirtualRace global for onclick handlers
window.VirtualRace = VirtualRace;

// Setup Virtual Race button
document.getElementById('virtualRaceBtn')?.addEventListener('click', () => VirtualRace.open());

// Setup mode tab clicks
document.querySelectorAll('.vr-mode-tab').forEach(tab => {
    tab.addEventListener('click', () => VirtualRace.setMode(tab.dataset.mode));
});

// Setup athlete select dropdowns
document.getElementById('vrSelectA')?.addEventListener('change', (e) => VirtualRace.selectAthlete('A', e.target.value));
document.getElementById('vrSelectB')?.addEventListener('change', (e) => VirtualRace.selectAthlete('B', e.target.value));

// Setup playback controls
document.getElementById('vrPlayBtn')?.addEventListener('click', () => VirtualRace.togglePlay());
document.querySelectorAll('.vr-speed-btn').forEach(btn => {
    btn.addEventListener('click', () => VirtualRace.setSpeed(parseFloat(btn.dataset.speed)));
});


// ── Boot ─────────────────────────────────────────────────────────────────
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
