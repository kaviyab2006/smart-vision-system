const statusText = document.getElementById('status-text');
const btnCamera = document.getElementById('btn-camera');
const fileUpload = document.getElementById('file-upload');
const btnVoice = document.getElementById('btn-voice');
const btnStop = document.getElementById('btn-stop');

const videoWrapper = document.getElementById('video-wrapper');
const videoFeed = document.getElementById('video-feed');
const outputCanvas = document.getElementById('output-canvas');
const outCtx = outputCanvas.getContext('2d');

const imageWrapper = document.getElementById('image-wrapper');
const imagePreview = document.getElementById('image-preview');
const staticCanvas = document.getElementById('static-canvas');
const staticCtx = staticCanvas.getContext('2d');

const BACKEND_URL = 'http://127.0.0.1:8000';

let cameraStream = null;
let processingInterval = null;
let isProcessing = false;
let lookingForObject = null; // Used when voice command says "Find wallet"

let synth = window.speechSynthesis;
let SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript.toLowerCase();
        updateStatus(`Heard: "${transcript}"`);
        
        // Very basic simple intent parsing
        if (transcript.includes('find') || transcript.includes('where is')) {
            const words = transcript.split(' ');
            const obj = words[words.length - 1]; // naive: pick last word
            lookingForObject = obj.replace(/[^\w\s]/gi, ''); // clean punctuation
            speak(`I will look for your ${lookingForObject}`);
        } else if (transcript.includes('stop')) {
            stopCamera();
        } else {
            speak("Sorry, I didn't catch that command.");
        }
    };
    
    recognition.onerror = (event) => {
        updateStatus(`Voice recognition error: ${event.error}`);
    };
}

function speak(text) {
    if (synth.speaking) {
        // Optionally cancel or queue
        // synth.cancel();
    }
    const msg = new SpeechSynthesisUtterance(text);
    msg.rate = 1.0;
    msg.pitch = 1.0;
    synth.speak(msg);
    updateStatus(text);
}

function updateStatus(text) {
    statusText.innerText = text;
}

// Draw bounding boxes
function drawBoxes(ctx, canvas, objects, width, height) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scale between real image and canvas
    const scaleX = canvas.width / width;
    const scaleY = canvas.height / height;
    
    objects.forEach(obj => {
        // Highlight logic
        const isTarget = lookingForObject && obj.name.toLowerCase().includes(lookingForObject);
        
        ctx.strokeStyle = isTarget ? '#00FF00' : '#ffff00';
        ctx.lineWidth = isTarget ? 6 : 3;
        ctx.fillStyle = ctx.strokeStyle;
        
        const x = obj.box.x1 * scaleX;
        const y = obj.box.y1 * scaleY;
        const w = (obj.box.x2 - obj.box.x1) * scaleX;
        const h = (obj.box.y2 - obj.box.y1) * scaleY;
        
        ctx.strokeRect(x, y, w, h);
        
        // Draw label
        ctx.font = '18px Inter, sans-serif';
        const text = `${obj.name} ${Math.round(obj.confidence * 100)}%`;
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillRect(x, y - 24, textWidth + 10, 24);
        ctx.fillStyle = '#000';
        ctx.fillText(text, x + 5, y - 6);
    });
}

async function sendFrameToAPI(blob, isVideo) {
    if (isProcessing) return;
    isProcessing = true;
    updateStatus("Processing...");
    
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');
    
    try {
        const response = await fetch(`${BACKEND_URL}/detect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("API Network error");
        
        const data = await response.json();
        
        if (data.hints && data.hints.length > 0) {
            // Filter hints if looking for specific object
            let relevantHints = data.hints;
            if (lookingForObject) {
                relevantHints = data.hints.filter(h => h.toLowerCase().includes(lookingForObject));
            }
            
            if (relevantHints.length > 0) {
                // If it's continuous video, we don't want to spam the voice
                if (isVideo && synth.speaking) {
                    // Skip if currently talking
                } else {
                    speak(relevantHints[0]); // Just say the first one to avoid overwhelming
                }
            } else if (lookingForObject) {
                // Not found
                if (!isVideo) speak(`I could not find your ${lookingForObject} here.`);
            }
        } else {
            if (!isVideo) speak("No recognizable objects detected.");
        }
        
        return data; // To let caller draw boxes
    } catch (e) {
        console.error(e);
        updateStatus("Error communicating with backend.");
    } finally {
        isProcessing = false;
    }
}

// Camera Upload logic
btnCamera.addEventListener('click', async () => {
    stopCamera();
    imageWrapper.classList.add('hidden');
    videoWrapper.classList.remove('hidden');
    btnStop.classList.remove('hidden');
    
    speak("Starting camera.");
    
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        videoFeed.srcObject = cameraStream;
        
        videoFeed.onloadedmetadata = () => {
            outputCanvas.width = videoFeed.videoWidth;
            outputCanvas.height = videoFeed.videoHeight;
        };
        
        // Every 3 seconds capture a frame
        processingInterval = setInterval(() => {
            if (videoFeed.readyState === videoFeed.HAVE_ENOUGH_DATA) {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = videoFeed.videoWidth;
                tempCanvas.height = videoFeed.videoHeight;
                tempCanvas.getContext('2d').drawImage(videoFeed, 0, 0);
                tempCanvas.toBlob(blob => {
                    sendFrameToAPI(blob, true).then(data => {
                        if (data && data.objects) {
                            window.requestAnimationFrame(() => {
                                drawBoxes(outCtx, outputCanvas, data.objects, videoFeed.videoWidth, videoFeed.videoHeight);
                            });
                        }
                    });
                }, 'image/jpeg', 0.8);
            }
        }, 3000);
        
    } catch (err) {
        updateStatus("Camera access denied or unavailable.");
        speak("Camera is not available.");
    }
});

btnStop.addEventListener('click', stopCamera);

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    videoWrapper.classList.add('hidden');
    btnStop.classList.add('hidden');
    outCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    updateStatus("Camera stopped.");
}

// File Upload logic
// Allow triggering file upload with keyboard
fileUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    stopCamera();
    videoWrapper.classList.add('hidden');
    imageWrapper.classList.remove('hidden');
    
    speak("Analyzing image.");
    
    const url = URL.createObjectURL(file);
    imagePreview.src = url;
    
    imagePreview.onload = () => {
        staticCanvas.width = imagePreview.naturalWidth;
        staticCanvas.height = imagePreview.naturalHeight;
        
        sendFrameToAPI(file, false).then(data => {
            if (data && data.objects) {
                drawBoxes(staticCtx, staticCanvas, data.objects, imagePreview.naturalWidth, imagePreview.naturalHeight);
            }
        });
    };
});

// Voice Command
btnVoice.addEventListener('click', () => {
    if (!recognition) {
        speak("Voice recognition is not supported in this browser.");
        return;
    }
    speak("Listening. Try asking to find your wallet.");
    // Small delay to let synthesis finish before listening
    setTimeout(() => {
         try {
             recognition.start();
         } catch(e) {}
    }, 2500);
});
