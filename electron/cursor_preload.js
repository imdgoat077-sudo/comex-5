const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('cursorNative', {
  onAnimate: (callback) => {
    ipcRenderer.removeAllListeners('cursor-animate');
    ipcRenderer.on('cursor-animate', (_event, data) => callback(data));
  },
  onReset: (callback) => {
    ipcRenderer.removeAllListeners('cursor-reset');
    ipcRenderer.on('cursor-reset', () => callback());
  }
});
