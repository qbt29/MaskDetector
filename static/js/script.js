const video = document.getElementById('video');
const statusDiv = document.getElementById('status');
const statsContent = document.getElementById('statsContent');
const camList = document.getElementById('camList');
const logsPanel = document.getElementById('logsPanel');
const logsContent = document.getElementById('logsContent');
const logsCount = document.getElementById('logsCount');

let logEntries = [];
const MAX_LOGS = 200;

let activeCameraID = -1;
video.src = `/api/video_feed/${activeCameraID}`;

// Перехватываем console.log/warn/error
const originalLog = console.log;
const originalWarn = console.warn;
const originalError = console.error;

console.log = (...args) => {
    originalLog(...args);
    addLog(args.join(' '), 'info');
};
console.warn = (...args) => {
    originalWarn(...args);
    addLog(args.join(' '), 'warn');
};
console.error = (...args) => {
    originalError(...args);
    addLog(args.join(' '), 'error');
};

function addLog(message, level = 'info') {
    const now = new Date().toLocaleTimeString();
    const entry = `[${now}] ${message}`;
    logEntries.push({text: entry, level});

    // Ограничиваем размер
    if (logEntries.length > MAX_LOGS) {
        logEntries.shift();
    }

    // Обновляем отображение, только если панель открыта
    if (logsPanel.style.display === 'block') {
        renderLogs();
    }
    logsCount.textContent = `(${logEntries.length})`;
}

function renderLogs() {
    logsContent.innerHTML = logEntries.map(item =>
        `<div class="log-entry log-${item.level}">${item.text}</div>`
    ).join('');
    logsContent.scrollTop = logsContent.scrollHeight;
}

function toggleLogs() {
    if (logsPanel.style.display === 'block') {
        logsPanel.style.display = 'none';
    } else {
        logsPanel.style.display = 'block';
        renderLogs();
    }
}

function showMessage(text, type = 'info') {
    statusDiv.textContent = text;
    statusDiv.className = `status ${type} show`;
    setTimeout(() => {
        statusDiv.classList.remove('show');
    }, 3000);
    // Добавляем в логи
    addLog(`UI: ${text}`, type === 'error' ? 'error' : 'info');
}

// ... остальные функции (switchCamera, takeScreenshot и т.д.) без изменений ...

function switchCamera(id) {
    activeCameraID = id;
    video.src = `/api/video_feed/${activeCameraID.toString()}`;
    console.log(`video source: ${video.src}, cameraID: ${activeCameraID}`);
    showMessage(`Камера переключена на ${activeCameraID}`);
}

async function listCameras(filter = 0) {
    const listJSON = await fetch('/api/list_cameras');
    const list = await listJSON.json();
    if (list.cameras.length == 0) {
        camList.innerHTML = '<option class="cam-btn" value="-1"">Выбор камеры:</option>';
        return;
    } else {
        if (activeCameraID < 0) {
            switchCamera(list.cameras[0].id);
        }
        camList.className = "cam-active";
        camList.innerHTML = `<option class="cam-active" value="${activeCameraID}">Камера ${activeCameraID}</option>`;

    }
    list.cameras.forEach(cam => {
        if (cam.isAlive > filter || cam.id == activeCameraID)
            return;
        // <button className="cam-btn" onClick="switchCamera(0)"> Камера 0</button>
        const el = document.createElement('option');
        el.className = "cam-btn";
        el.onClick = () => {
            switchCamera(cam.id)
        };
        el.value = cam.id;
        el.textContent = `Камера ${cam.id}`;
        camList.appendChild(el);
    })
}

async function onSelectCamIDChange(event) {
    switchCamera(event.target.value);
}

async function takeScreenshot() {
    if (activeCameraID < 0) {
        showMessage('Нет активной камеры', 'error')
        return;
    }
    try {
        const link = document.createElement('a');
        link.href = `/api/current_frame/${activeCameraID}`;
        link.download = `mask_screenshot_${activeCameraID}_${Date.now()}.jpg`;
        link.click();
        showMessage('Скриншот сохранён', 'success');
    } catch (e) {
        showMessage('Ошибка скриншота', 'error');
        console.log(e);
    }
}

async function resetTracks() {
    try {
        const res = await fetch(`/api/reset_tracks/${activeCameraID}`, {method: 'POST'});
        const data = await res.json();
        if (data.status === 'success') {
            showMessage('Треки сброшены', 'info');
        }
    } catch (e) {
        showMessage('Не удалось сбросить треки', 'error');
    }
}

// Инициализация
window.addEventListener('load', () => {
    addLog('Страница загружена', 'info');
    listCameras(0);
});