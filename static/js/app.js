/**
 * MaskFace Frontend Controller
 * Implements real-time webcam frame streaming, Web Audio API chime with 5-frame persistence debouncing,
 * and numbered bounding box audit synchronization.
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Tab Switching Logic
    // -------------------------------------------------------------------------
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const activePane = document.getElementById(target);
            if (activePane) activePane.classList.add("active");
        });
    });

    // -------------------------------------------------------------------------
    // Hardware Status Synchronization (GPU vs CPU)
    // -------------------------------------------------------------------------
    const hwBadge = document.getElementById("hardware-badge");
    async function updateHardwareStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                if (hwBadge) {
                    hwBadge.textContent = "Device: " + data.active_device;
                    hwBadge.className = "badge badge-device";
                }
            }
        } catch (err) {
            console.warn("Status check error:", err);
        }
    }
    updateHardwareStatus();

    // -------------------------------------------------------------------------
    // 2. Slider Value Sync & Configuration Updates
    // -------------------------------------------------------------------------
    const detConfSlider = document.getElementById("det-conf-slider");
    const detConfVal = document.getElementById("det-conf-val");
    const classConfSlider = document.getElementById("class-conf-slider");
    const classConfVal = document.getElementById("class-conf-val");
    const marginSlider = document.getElementById("margin-slider");
    const marginVal = document.getElementById("margin-val");
    const resSlider = document.getElementById("res-slider");
    const resVal = document.getElementById("res-val");
    const tempSlider = document.getElementById("temp-slider");
    const tempVal = document.getElementById("temp-val");

    detConfSlider.addEventListener("input", () => detConfVal.textContent = detConfSlider.value);
    classConfSlider.addEventListener("input", () => classConfVal.textContent = classConfSlider.value);
    marginSlider.addEventListener("input", () => marginVal.textContent = marginSlider.value);
    resSlider.addEventListener("input", () => resVal.textContent = `${resSlider.value}px`);
    tempSlider.addEventListener("input", () => tempVal.textContent = tempSlider.value);

    let currentStaticSource = { type: "preset", sample: "maksssksksss1.png" };

    async function reanalyzeStaticImage() {
        if (!currentStaticSource) return;
        try {
            if (currentStaticSource.type === "preset") {
                const res = await fetch("/api/predict-image", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_name: currentStaticSource.sample })
                });
                if (res.ok) {
                    const data = await res.json();
                    renderStaticResults(data);
                }
            } else if (currentStaticSource.type === "file") {
                const formData = new FormData();
                formData.append("image", currentStaticSource.file);
                const res = await fetch("/api/predict-image", {
                    method: "POST",
                    body: formData
                });
                if (res.ok) {
                    const data = await res.json();
                    renderStaticResults(data);
                }
            }
        } catch (err) {
            console.error("Re-analysis failed:", err);
        }
    }

    async function sendPipelineConfig() {
        const policyElem = document.querySelector('input[name="policy"]:checked');
        const selectedPolicy = policyElem ? policyElem.value : "0.25";
        const configPayload = {
            model: document.getElementById("model-select").value,
            detector: document.getElementById("detector-select").value,
            detection_conf: parseFloat(detConfSlider.value),
            min_compliance_conf: parseFloat(classConfSlider.value),
            min_margin: parseFloat(marginSlider.value),
            min_bbox_size: parseInt(resSlider.value),
            temperature: parseFloat(tempSlider.value),
            improper_weight: parseFloat(selectedPolicy)
        };

        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(configPayload)
            });
            if (res.ok) {
                await reanalyzeStaticImage();
                return true;
            }
        } catch (err) {
            console.error("Config update failed:", err);
        }
        return false;
    }

    const applyConfigBtn = document.getElementById("apply-config-btn");
    applyConfigBtn.addEventListener("click", async () => {
        applyConfigBtn.textContent = "Applying...";
        const ok = await sendPipelineConfig();
        if (ok) {
            applyConfigBtn.textContent = "Saved";
            setTimeout(() => applyConfigBtn.textContent = "Apply Settings", 1500);
        } else {
            applyConfigBtn.textContent = "Error";
            setTimeout(() => applyConfigBtn.textContent = "Apply Settings", 1500);
        }
    });

    // Auto-update when dropdowns or policies change
    document.getElementById("model-select").addEventListener("change", sendPipelineConfig);
    document.getElementById("detector-select").addEventListener("change", sendPipelineConfig);
    document.querySelectorAll('input[name="policy"]').forEach(r => r.addEventListener("change", sendPipelineConfig));

    // Auto-update when sliders finish dragging
    [detConfSlider, classConfSlider, marginSlider, resSlider, tempSlider].forEach(slider => {
        slider.addEventListener("change", sendPipelineConfig);
    });

    // -------------------------------------------------------------------------
    // 3. Tab 1: Static Image Inspection & Numbered Audit Table
    // -------------------------------------------------------------------------
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const presetButtons = document.querySelectorAll(".btn-preset");

    const staticOrigImg = document.getElementById("static-orig-img");
    const staticAnnotatedImg = document.getElementById("static-annotated-img");
    const origPlaceholder = document.getElementById("orig-placeholder");
    const annotatedPlaceholder = document.getElementById("annotated-placeholder");

    const staticTotal = document.getElementById("static-total-faces");
    const staticSafe = document.getElementById("static-safe-faces");
    const staticViol = document.getElementById("static-violations");
    const staticUnc = document.getElementById("static-uncertain");
    const staticScore = document.getElementById("static-safety-index");
    const staticLatency = document.getElementById("static-latency");
    const staticAlert = document.getElementById("static-alert-banner");
    const staticTableBody = document.getElementById("static-table-body");

    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadAndAnalyze(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#38bdf8";
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadAndAnalyze(e.dataTransfer.files[0]);
        }
    });

    presetButtons.forEach(btn => {
        btn.addEventListener("click", async () => {
            presetButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const sampleName = btn.dataset.sample;
            currentStaticSource = { type: "preset", sample: sampleName };
            try {
                const res = await fetch("/api/predict-image", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_name: sampleName })
                });
                if (res.ok) {
                    const data = await res.json();
                    renderStaticResults(data);
                }
            } catch (err) {
                console.error("Failed to load preset:", err);
            }
        });
    });

    async function uploadAndAnalyze(file) {
        currentStaticSource = { type: "file", file: file };
        const formData = new FormData();
        formData.append("image", file);

        try {
            const res = await fetch("/api/predict-image", {
                method: "POST",
                body: formData
            });
            if (res.ok) {
                const data = await res.json();
                renderStaticResults(data);
            }
        } catch (err) {
            console.error("Upload analysis failed:", err);
        }
    }

    function renderStaticResults(data) {
        if (!data || !data.summary) return;

        const summary = data.summary;
        staticTotal.textContent = summary.total_faces;
        
        const safePct = summary.total_faces > 0 ? Math.round((summary.safe_count / summary.total_faces) * 100) : 100;
        staticSafe.innerHTML = `${summary.safe_count} <span style="font-size: 0.95rem; color: #94a3b8;">(${safePct}%)</span>`;
        staticViol.textContent = summary.violation_count;
        staticUnc.textContent = summary.uncertain_count;
        staticScore.textContent = `${summary.frame_compliance_score}%`;
        staticLatency.textContent = `(${data.latency_ms}ms)`;

        // Image displays
        staticOrigImg.src = data.original_image;
        staticOrigImg.classList.remove("hidden");
        origPlaceholder.classList.add("hidden");

        staticAnnotatedImg.src = data.annotated_image;
        staticAnnotatedImg.classList.remove("hidden");
        annotatedPlaceholder.classList.add("hidden");

        // Alert banner
        if (summary.violation_count > 0) {
            staticAlert.className = "alert-banner danger";
            staticAlert.innerHTML = `<strong>Violation Alert:</strong> ${summary.violation_count} non-compliant individual(s) detected.`;
            staticAlert.classList.remove("hidden");
        } else if (summary.uncertain_count > 0) {
            staticAlert.className = "alert-banner";
            staticAlert.style.borderLeft = "3px solid #f59e0b";
            staticAlert.style.background = "rgba(245, 158, 11, 0.08)";
            staticAlert.innerHTML = `<strong>Verification Advisory:</strong> ${summary.uncertain_count} face crop(s) flagged as uncertain by rejection gates.`;
            staticAlert.classList.remove("hidden");
        } else {
            staticAlert.className = "alert-banner safe";
            staticAlert.innerHTML = `<strong>Status Compliant:</strong> All detected individuals satisfy proper mask safety standards.`;
            staticAlert.classList.remove("hidden");
        }

        // Populate Numbered Table
        staticTableBody.innerHTML = "";
        if (data.scored_faces && data.scored_faces.length > 0) {
            data.scored_faces.forEach(face => {
                const tr = document.createElement("tr");
                const bbox = face.bbox;
                const w = bbox[2] - bbox[0];
                const h = bbox[3] - bbox[1];
                const audit = face.audit_trail;

                tr.innerHTML = `
                    <td><span class="face-id-badge">#${face.face_id || 1}</span></td>
                    <td><span class="status-badge ${face.status}">${face.status}</span></td>
                    <td><strong>${face.display_label}</strong></td>
                    <td>${face.confidence}%</td>
                    <td>${face.margin}%</td>
                    <td><strong>${face.compliance_score}/100</strong></td>
                    <td>${w} × ${h} px</td>
                    <td>${audit.rejection_gate || "Cleared"}</td>
                    <td class="text-secondary" style="font-size: 0.8rem;">${face.reason}</td>
                `;
                staticTableBody.appendChild(tr);
            });
        } else {
            staticTableBody.innerHTML = `<tr><td colspan="9" class="text-center text-muted">No faces detected. Try lowering the detection confidence slider.</td></tr>`;
        }
    }

    // Load initial crowd sample on boot
    presetButtons[0].click();

    // -------------------------------------------------------------------------
    // 4. Tab 2: Live Video, Webcam Streaming & 5-Frame Persistence Audio Alarm
    // -------------------------------------------------------------------------
    const btnModeWebcam = document.getElementById("btn-mode-webcam");
    const btnModeCctv = document.getElementById("btn-mode-cctv");
    const webcamContainer = document.getElementById("webcam-container");
    const cctvContainer = document.getElementById("cctv-container");
    const cctvImg = document.getElementById("cctv-mjpeg-img");

    btnModeWebcam.addEventListener("click", () => {
        btnModeWebcam.classList.add("active");
        btnModeCctv.classList.remove("active");
        webcamContainer.classList.remove("hidden");
        cctvContainer.classList.add("hidden");
        cctvImg.src = "";
    });

    btnModeCctv.addEventListener("click", () => {
        btnModeCctv.classList.add("active");
        btnModeWebcam.classList.remove("active");
        cctvContainer.classList.remove("hidden");
        webcamContainer.classList.add("hidden");
        cctvImg.src = "/api/video-feed";
    });

    // Web Audio API Synthesizer
    let audioCtx = null;
    let audioEnabled = true;
    const toggleAudioBtn = document.getElementById("toggle-audio-btn");

    toggleAudioBtn.addEventListener("click", () => {
        audioEnabled = !audioEnabled;
        toggleAudioBtn.textContent = audioEnabled ? "Alarm: Active" : "Alarm: Muted";
        toggleAudioBtn.className = audioEnabled ? "btn btn-secondary" : "btn btn-danger";
    });

    let lastChimeTime = 0;
    const CHIME_INTERVAL_MS = 900; // Alert chime repeats at most once every 900ms during sustained violation

    function playSecurityChime() {
        if (!audioEnabled) return;
        const now = Date.now();
        if (now - lastChimeTime < CHIME_INTERVAL_MS) return;
        lastChimeTime = now;

        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === "suspended") {
                audioCtx.resume();
            }

            // Dual-frequency security chime (880Hz -> 440Hz alert)
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(880, audioCtx.currentTime); // High pitch
            osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.20); // Drop

            gain.gain.setValueAtTime(0.20, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.30);

            osc.connect(gain);
            gain.connect(audioCtx.destination);

            osc.start();
            osc.stop(audioCtx.currentTime + 0.30);
        } catch (e) {
            console.warn("Audio chime failed:", e);
        }
    }

    // Webcam Video & Canvas Loop
    const startWebcamBtn = document.getElementById("start-webcam-btn");
    const stopWebcamBtn = document.getElementById("stop-webcam-btn");
    const videoElem = document.getElementById("webcam-video");
    const canvasElem = document.getElementById("webcam-canvas");
    const ctx = canvasElem.getContext("2d");

    const liveRollingScore = document.getElementById("live-rolling-score");
    const liveSafeCount = document.getElementById("live-safe-count");
    const liveViolCount = document.getElementById("live-viol-count");
    const liveFps = document.getElementById("live-fps");
    const persistenceDots = document.querySelectorAll("#persistence-dots .dot");

    let isStreaming = false;
    let mediaStream = null;
    let offscreenCanvas = document.createElement("canvas");
    let offscreenCtx = offscreenCanvas.getContext("2d");

    startWebcamBtn.addEventListener("click", async () => {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });
            videoElem.srcObject = mediaStream;
            await videoElem.play();

            canvasElem.width = 640;
            canvasElem.height = 480;
            offscreenCanvas.width = 640;
            offscreenCanvas.height = 480;

            isStreaming = true;
            startWebcamBtn.disabled = true;
            stopWebcamBtn.disabled = false;

            processWebcamLoop();
        } catch (err) {
            alert(`Could not access webcam: ${err.message}. Please check browser permissions.`);
        }
    });

    stopWebcamBtn.addEventListener("click", () => {
        isStreaming = false;
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
        startWebcamBtn.disabled = false;
        stopWebcamBtn.disabled = true;
        ctx.clearRect(0, 0, canvasElem.width, canvasElem.height);
    });

    async function processWebcamLoop() {
        if (!isStreaming) return;

        // Draw video frame to offscreen canvas
        offscreenCtx.drawImage(videoElem, 0, 0, 640, 480);
        const frameBase64 = offscreenCanvas.toDataURL("image/jpeg", 0.75);

        try {
            const res = await fetch("/api/process-frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: frameBase64 })
            });

            if (res.ok) {
                const data = await res.json();
                
                // Draw annotated frame onto main visible canvas
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, 640, 480);
                };
                img.src = data.annotated_frame;

                // Update Telemetry
                const summary = data.summary;
                liveRollingScore.textContent = `${summary.rolling_compliance_score}%`;
                liveSafeCount.textContent = summary.safe_count;
                liveViolCount.textContent = summary.violation_count;
                liveFps.textContent = `${data.fps} FPS`;

                // Update 5-Frame Persistence Gauge
                const streak = Math.min(5, data.violation_streak || 0);
                persistenceDots.forEach((dot, idx) => {
                    if (idx < streak) {
                        dot.classList.add("active");
                    } else {
                        dot.classList.remove("active");
                    }
                });

                // Play Chime ONLY when 5-frame persistence is reached
                if (data.alarm_triggered) {
                    playSecurityChime();
                } else {
                    lastChimeTime = 0; // Instantly ready for next violation, completely silent now
                }
            }
        } catch (err) {
            console.error("Frame processing error:", err);
        }

        if (isStreaming) {
            requestAnimationFrame(processWebcamLoop);
        }
    }
});
