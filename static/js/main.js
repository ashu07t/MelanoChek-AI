// Main JavaScript for Skin Cancer Detection App

document.addEventListener('DOMContentLoaded', function() {

// ==================== ADD THESE NEW VARIABLES AT THE TOP ====================
// Camera functionality variables
let cameraStream = null;
let capturedImageData = null;
let patientData = null;
// ============================================================================

// Get DOM elements (YOUR EXISTING CODE)
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');

// File upload event listeners (YOUR EXISTING CODE)
imageInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleFileDrop);

// ==================== ADD THESE NEW EVENT LISTENERS ====================
// Camera button event listeners
document.getElementById('startCameraBtn').addEventListener('click', startCamera);
document.getElementById('capturePhotoBtn').addEventListener('click', capturePhoto);
document.getElementById('stopCameraBtn').addEventListener('click', stopCamera);
document.getElementById('downloadReportBtn').addEventListener('click', downloadReport);
// ======================================================================

// Prevent default drag behaviors (YOUR EXISTING CODE)
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragOver(e) {
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    uploadArea.classList.remove('dragover');
}

function handleFileDrop(e) {
    uploadArea.classList.remove('dragover');
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const file = imageInput.files[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showAlert('Please select a valid image file (PNG, JPG, JPEG, BMP, TIFF)', 'danger');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'danger');
        return;
    }

    // ==================== REPLACE THIS LINE ====================
    // OLD: uploadAndPredict(file);
    // NEW: Use the new function that validates patient info
    uploadAndAnalyze(file);
    // ===========================================================
}

// ==================== ADD THESE NEW CAMERA FUNCTIONS ====================
async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment' // Use rear camera on mobile
            }
        };
        
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('cameraVideo');
        video.srcObject = cameraStream;
        
        document.getElementById('cameraContainer').style.display = 'block';
        document.getElementById('capturePhotoBtn').style.display = 'inline-block';
        document.getElementById('stopCameraBtn').style.display = 'inline-block';
        document.getElementById('startCameraBtn').style.display = 'none';
    } catch (error) {
        showAlert('Camera access denied or not available: ' + error.message, 'danger');
        console.error('Camera error:', error);
    }
}

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('captureCanvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    // Convert to blob and create file
    canvas.toBlob(blob => {
        const file = new File([blob], 'captured_photo.jpg', { type: 'image/jpeg' });
        
        // Trigger upload with captured image
        uploadAndAnalyze(file);
        stopCamera();
    }, 'image/jpeg', 0.95);
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    document.getElementById('cameraContainer').style.display = 'none';
    document.getElementById('capturePhotoBtn').style.display = 'none';
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('startCameraBtn').style.display = 'inline-block';
}
// ========================================================================

// ==================== REPLACE uploadAndPredict WITH THIS ====================
async function uploadAndAnalyze(file) {
    // Validate patient information
    const name = document.getElementById('patientName').value.trim();
    const age = document.getElementById('patientAge').value;
    const gender = document.getElementById('patientGender').value;
    
    if (!name || !age || !gender) {
        showAlert('Please fill in all patient information before uploading or capturing an image.', 'warning');
        return;
    }
    
    // Store patient data
    patientData = { name, age, gender };
    
    // Show loading indicator
    showLoading(true);
    hideResults();

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send request to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            showAlert(result.error, 'danger');
        } else {
            // Store results for report
            capturedImageData = result;
            displayResults(result);
        }

    } catch (error) {
        console.error('Error:', error);
        showAlert('An error occurred during prediction. Please try again.', 'danger');
    } finally {
        showLoading(false);
    }
}
// ============================================================================

function displayResults(result) {
    // Display uploaded image
    const uploadedImage = document.getElementById('uploadedImage');
    uploadedImage.src = result.image_url;

    // Display prediction results
    document.getElementById('predictedClass').textContent = result.predicted_class;
    document.getElementById('confidence').textContent = `${result.confidence.toFixed(1)}%`;

    const riskElement = document.getElementById('riskLevel');
    riskElement.textContent = result.risk_level;

    // Style risk level based on severity
    riskElement.className = '';
    if (result.risk_level.includes('High Risk')) {
        riskElement.classList.add('risk-high');
    } else if (result.risk_level.includes('Medium Risk')) {
        riskElement.classList.add('risk-medium');
    } else {
        riskElement.classList.add('risk-low');
    }

    // Display probability chart
    displayProbabilityChart(result.all_probabilities);

    // Show results section
    showResults();
}

function displayProbabilityChart(probabilities) {
    const chartContainer = document.getElementById('probabilityChart');
    chartContainer.innerHTML = '';

    // Sort probabilities in descending order
    const sortedProbs = Object.entries(probabilities)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5); // Show top 5

    sortedProbs.forEach(([className, probability]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar';

        const label = document.createElement('div');
        label.className = 'probability-label d-flex justify-content-between';
        label.innerHTML = `
            <span>${className.replace(/\([^)]*\)/g, '').trim()}</span>
            <span>${probability.toFixed(1)}%</span>
        `;

        const progressBar = document.createElement('div');
        progressBar.className = 'progress mt-1';
        progressBar.style.height = '8px';

        const progressFill = document.createElement('div');
        progressFill.className = 'progress-bar bg-primary';
        progressFill.style.width = `${probability}%`;

        progressBar.appendChild(progressFill);
        barContainer.appendChild(label);
        barContainer.appendChild(progressBar);
        chartContainer.appendChild(barContainer);
    });
}

// ==================== ADD THIS NEW PDF DOWNLOAD FUNCTION ====================
async function downloadReport() {
    if (!capturedImageData || !patientData) {
        showAlert('No detection results available to download.', 'warning');
        return;
    }
    
    // Call backend to generate PDF
    const reportData = {
        patient: patientData,
        results: capturedImageData
    };
    
    try {
        const response = await fetch('/generate-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reportData)
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Skin_Cancer_Report_${patientData.name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showAlert('Report downloaded successfully!', 'success');
        } else {
            showAlert('Failed to generate report.', 'danger');
        }
    } catch (error) {
        console.error('Download error:', error);
        showAlert('Error downloading report: ' + error.message, 'danger');
    }
}
// ============================================================================

function showLoading(show) {
    loadingIndicator.style.display = show ? 'block' : 'none';
}

function showResults() {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show mt-3`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert after upload area
    uploadArea.parentNode.insertBefore(alertDiv, uploadArea.nextSibling);

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Smooth scroll for navigation links (YOUR EXISTING CODE)
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            const offsetTop = targetElement.offsetTop - 70; // Account for fixed navbar
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    });
});

}); // End of DOMContentLoaded
