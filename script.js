class MNISTClassifier {
    constructor() {
        this.session = null;
        this.isDrawing = false;
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.clearBtn = document.getElementById('clearBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.prediction = document.getElementById('prediction');
        this.confidence = document.getElementById('confidence');
        this.probabilities = document.getElementById('probabilities');

        this.setupCanvas();
        this.setupButtons();
        this.loadModel();
    }

    setupCanvas() {
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 15;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());

        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            this.canvas.dispatchEvent(mouseEvent);
        });
    }

    setupButtons() {
        this.clearBtn.addEventListener('click', () => this.clearCanvas());
        this.predictBtn.addEventListener('click', () => this.predict());
    }

    async loadModel() {
        try {
            this.prediction.textContent = 'Chargement du modèle...';
            this.prediction.className = 'prediction loading';

            try {
                this.session = await ort.InferenceSession.create('./mnist_model_improved.onnx');
                console.log('Modèle amélioré chargé');
            } catch (error) {
                console.log('Modèle amélioré non trouvé, utilisation du modèle original');
                this.session = await ort.InferenceSession.create('./mnist_model.onnx');
            }

            this.prediction.textContent = 'Modèle chargé ! Dessinez un chiffre sur le canvas';
            this.prediction.className = 'prediction';
            this.predictBtn.disabled = false;

            console.log('Modèle ONNX chargé avec succès');
        } catch (error) {
            console.error('Erreur lors du chargement du modèle:', error);
            this.prediction.textContent = 'Erreur: Impossible de charger le modèle';
            this.prediction.className = 'prediction';
        }
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.ctx.beginPath();
        this.ctx.moveTo(pos.x, pos.y);
    }

    draw(e) {
        if (!this.isDrawing) return;

        const pos = this.getMousePos(e);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    clearCanvas() {
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.prediction.textContent = 'Dessinez un chiffre sur le canvas';
        this.confidence.textContent = '';
        this.probabilities.innerHTML = '';
    }

    preprocessCanvas() {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;

        tempCtx.drawImage(this.canvas, 0, 0, 28, 28);

        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;

        const input = new Float32Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            const pixelIndex = i * 4;
            const grayscale = (data[pixelIndex] + data[pixelIndex + 1] + data[pixelIndex + 2]) / 3;
            input[i] = (grayscale / 255.0 - 0.1307) / 0.3081;
        }

        return input;
    }

    async predict() {
        if (!this.session) {
            this.prediction.textContent = 'Modèle non chargé';
            return;
        }

        try {
            this.prediction.textContent = 'Prédiction en cours...';
            this.prediction.className = 'prediction loading';

            const inputData = this.preprocessCanvas();
            const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);

            const results = await this.session.run({ input: inputTensor });
            const output = results.output.data;

            const probabilities = this.softmax(Array.from(output));
            const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
            const confidence = (probabilities[predictedDigit] * 100).toFixed(1);

            this.prediction.textContent = `Prédiction: ${predictedDigit}`;
            this.prediction.className = 'prediction';
            this.confidence.textContent = `Confiance: ${confidence}%`;

            this.displayProbabilities(probabilities);

        } catch (error) {
            console.error('Erreur lors de la prédiction:', error);
            this.prediction.textContent = 'Erreur lors de la prédiction';
            this.prediction.className = 'prediction';
        }
    }

    softmax(arr) {
        const maxVal = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - maxVal));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sumExp);
    }

    displayProbabilities(probabilities) {
        this.probabilities.innerHTML = '';

        probabilities.forEach((prob, digit) => {
            const item = document.createElement('div');
            item.className = 'prob-item';

            const percentage = (prob * 100).toFixed(1);

            item.innerHTML = `
                <div class="prob-digit">${digit}</div>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="prob-value">${percentage}%</div>
            `;

            this.probabilities.appendChild(item);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MNISTClassifier();
});