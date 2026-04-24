/* ════════════════════════════════════════════════════════════════
   NeuroScan AI — Frontend Logic
   ════════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
    // ── DOM References ─────────────────────────────────────────
    const dropZone      = document.getElementById("dropZone");
    const dropContent   = document.getElementById("dropContent");
    const dropPreview   = document.getElementById("dropPreview");
    const previewImage  = document.getElementById("previewImage");
    const fileInput     = document.getElementById("fileInput");
    const browseBtn     = document.getElementById("browseBtn");
    const removeBtn     = document.getElementById("removeBtn");
    const analyzeBtn    = document.getElementById("analyzeBtn");
    const btnLoader     = document.getElementById("btnLoader");

    const uploadSection  = document.getElementById("uploadSection");
    const loadingSection = document.getElementById("loadingSection");
    const resultsSection = document.getElementById("resultsSection");

    // Result elements
    const resultBadge     = document.getElementById("resultBadge");
    const badgeIcon       = document.getElementById("badgeIcon");
    const resultLabel     = document.getElementById("resultLabel");
    const resultSubtext   = document.getElementById("resultSubtext");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceFill  = document.getElementById("confidenceFill");

    const originalImg = document.getElementById("originalImg");
    const contourImg  = document.getElementById("contourImg");
    const grayImg     = document.getElementById("grayImg");
    const threshImg   = document.getElementById("threshImg");
    const morphImg    = document.getElementById("morphImg");
    const cannyImg    = document.getElementById("cannyImg");

    const metricArea   = document.getElementById("metricArea");
    const metricRatio  = document.getElementById("metricRatio");
    const metricStatus = document.getElementById("metricStatus");

    // Volume Analyzer elements (analyzerTab)
    const vaVolume       = document.getElementById("vaVolume");
    const vaDiameter     = document.getElementById("vaDiameter");
    const vaRiskLevel    = document.getElementById("vaRiskLevel");
    const vaRiskIcon     = document.getElementById("vaRiskIcon");
    const vaGaugeFill    = document.getElementById("vaGaugeFill");
    const vaGaugeMarker  = document.getElementById("vaGaugeMarker");
    const chipSurgery    = document.getElementById("chipSurgery");
    const chipRadiation  = document.getElementById("chipRadiation");
    const chipChemo      = document.getElementById("chipChemo");
    const vaNoteText     = document.getElementById("vaNoteText");
    const analyzerEmptyState = document.getElementById("analyzerEmptyState");
    const analyzerResults    = document.getElementById("analyzerResults");

    const newScanBtn = document.getElementById("newScanBtn");

    let selectedFile = null;

    // ── Theme Toggle ───────────────────────────────────────────
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const moonIcon = document.getElementById("moonIcon");
    const sunIcon = document.getElementById("sunIcon");

    // Check for saved theme
    if (localStorage.getItem("theme") === "dark") {
        document.documentElement.classList.add("dark-mode");
        if (moonIcon) moonIcon.classList.add("hidden");
        if (sunIcon) sunIcon.classList.remove("hidden");
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            document.documentElement.classList.toggle("dark-mode");
            const isDark = document.documentElement.classList.contains("dark-mode");
            
            if (isDark) {
                if (moonIcon) moonIcon.classList.add("hidden");
                if (sunIcon) sunIcon.classList.remove("hidden");
                localStorage.setItem("theme", "dark");
            } else {
                if (sunIcon) sunIcon.classList.add("hidden");
                if (moonIcon) moonIcon.classList.remove("hidden");
                localStorage.setItem("theme", "light");
            }
        });
    }

    // ── Tabs Switching ─────────────────────────────────────────
    const tabs = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            document.getElementById(`${target}Tab`).classList.add("active");
        });
    });

    // "View Full Volume Analysis" button in results
    document.getElementById("goToVolumeBtn")?.addEventListener("click", () => {
        document.getElementById("tabAnalyzer").click();
    });


    // ── Drag & Drop ────────────────────────────────────────────
    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener("click", () => {
        if (!selectedFile) fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select an image file.");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            // Also pre-load the scanning viewport image
            const scanningImage = document.getElementById("scanningImage");
            if (scanningImage) scanningImage.src = e.target.result;
            dropContent.classList.add("hidden");
            dropPreview.classList.remove("hidden");
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        dropContent.classList.remove("hidden");
        dropPreview.classList.add("hidden");
        analyzeBtn.disabled = true;
    }

    // ── Analysis ───────────────────────────────────────────────
    analyzeBtn.addEventListener("click", () => {
        if (!selectedFile) return;
        startAnalysis();
    });

    async function startAnalysis() {
        // Show loading
        uploadSection.classList.add("hidden");
        loadingSection.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        // Reset HUD
        const scanPct    = document.getElementById("scanPct");
        const scanStatus = document.getElementById("scanStatus");
        if (scanPct)    scanPct.textContent    = "0%";
        if (scanStatus) scanStatus.textContent  = "SCANNING…";

        // Animate percentage counter
        let pct = 0;
        const pctTimer = setInterval(() => {
            pct = Math.min(pct + Math.floor(Math.random() * 4) + 1, 95);
            if (scanPct) scanPct.textContent = pct + "%";
        }, 180);

        // Animate loading steps
        animateLoadingSteps();

        // Send request
        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const res = await fetch("/analyze", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();

            if (data.error) {
                clearInterval(pctTimer);
                alert("Error: " + data.error);
                resetToUpload();
                return;
            }

            // Finish percentage to 100
            clearInterval(pctTimer);
            if (scanPct)    scanPct.textContent    = "100%";
            if (scanStatus) scanStatus.textContent  = "COMPLETE";

            // Let the loading animation finish
            await completeLoadingSteps();

            // Show results
            showResults(data);
        } catch (err) {
            clearInterval(pctTimer);
            console.error(err);
            alert("Failed to analyze the image. Please try again.");
            resetToUpload();
        }
    }

    function animateLoadingSteps() {
        const steps = ["step1", "step2", "step3", "step4"];
        steps.forEach((id) => {
            const el = document.getElementById(id);
            el.classList.remove("active", "done");
        });
        document.getElementById("step1").classList.add("active");
    }

    function completeLoadingSteps() {
        return new Promise((resolve) => {
            const steps = ["step1", "step2", "step3", "step4"];
            let i = 0;
            const interval = setInterval(() => {
                if (i > 0) {
                    document.getElementById(steps[i - 1]).classList.remove("active");
                    document.getElementById(steps[i - 1]).classList.add("done");
                }
                if (i < steps.length) {
                    document.getElementById(steps[i]).classList.add("active");
                } else {
                    clearInterval(interval);
                    setTimeout(resolve, 400);
                }
                i++;
            }, 500);
        });
    }

    // ── Show Results ───────────────────────────────────────────
    function showResults(data) {
        loadingSection.classList.add("hidden");
        resultsSection.classList.remove("hidden");

        const cls = data.classification;
        const det = data.detection;
        const base = data.upload_base;

        // Classification badge
        const isTumor = cls.has_tumor;
        resultBadge.className = "result-badge " + (isTumor ? "tumor" : "no-tumor");
        badgeIcon.textContent = isTumor ? "⚠" : "✓";
        resultLabel.textContent = isTumor ? "Tumor Detected" : "No Tumor Detected";
        resultSubtext.textContent = isTumor
            ? "Abnormality found in the MRI scan"
            : "Brain MRI appears normal";

        // Confidence
        confidenceValue.textContent = cls.confidence.toFixed(1) + "%";
        setTimeout(() => {
            confidenceFill.style.width = cls.confidence + "%";
        }, 100);

        // Images
        originalImg.src = base + det.original;
        contourImg.src  = base + det.contour;
        grayImg.src     = base + det.grayscale;
        threshImg.src   = base + det.threshold;
        morphImg.src    = base + det.morphology;
        cannyImg.src    = base + det.canny;

        // Metrics
        metricArea.textContent = det.tumor_area.toLocaleString();
        metricRatio.textContent = (det.tumor_ratio * 100).toFixed(3) + "%";
        metricStatus.textContent = det.tumor_found ? "Region Found" : "No Region";
        metricStatus.style.color = det.tumor_found ? "var(--danger)" : "var(--success)";

        // ── Volume Analyzer ─────────────────────────────────────────
        const va = data.volume_analysis;
        if (va) {
            // Show analyzer tab has data
            analyzerEmptyState.classList.add("hidden");
            analyzerResults.classList.remove("hidden");

            vaVolume.textContent   = va.estimated_volume_cm3 > 0 ? va.estimated_volume_cm3.toFixed(2) : "0";
            vaDiameter.textContent = va.estimated_diameter_cm > 0 ? va.estimated_diameter_cm.toFixed(2) : "0";
            vaRiskLevel.textContent = va.risk_level;

            // Risk icon colouring
            let riskColor;
            if      (va.risk_level === "Low")      riskColor = "var(--success)";
            else if (va.risk_level === "Moderate") riskColor = "var(--warning)";
            else if (va.risk_level === "High")     riskColor = "#f97316";
            else if (va.risk_level === "Critical") riskColor = "var(--danger)";
            else                                   riskColor = "var(--text-muted)";

            vaRiskLevel.style.color = riskColor;
            vaRiskIcon.querySelector("svg").style.stroke = riskColor;

            // Gauge
            setTimeout(() => {
                vaGaugeFill.style.width   = `${va.risk_score}%`;
                vaGaugeMarker.style.left  = `${va.risk_score}%`;
            }, 500);

            // Treatment chips
            chipSurgery.classList.toggle("active",   va.surgery_suitable);
            chipRadiation.classList.toggle("active",  va.radiation_suitable);
            chipChemo.classList.toggle("active",     va.chemo_suitable);

            vaNoteText.textContent = va.clinical_note;
        } else {
            // No volume data — keep analyzer tab showing empty state
            analyzerEmptyState.classList.remove("hidden");
            analyzerResults.classList.add("hidden");
        }

        // Scroll to top of results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ── Reset ──────────────────────────────────────────────────
    newScanBtn.addEventListener("click", () => {
        resetToUpload();
    });

    function resetToUpload() {
        clearFile();
        loadingSection.classList.add("hidden");
        resultsSection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
        confidenceFill.style.width = "0%";

        // Reset loading steps
        ["step1", "step2", "step3", "step4"].forEach((id) => {
            document.getElementById(id).classList.remove("active", "done");
        });
    }

    // Helper for number animation
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = (progress * (end - start) + start).toFixed(2);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
