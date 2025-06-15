// Establish a connection to the Flask-SocketIO server
const socket = io.connect('http://' + document.domain + ':' + location.port);

// When the server sends the current mode, update the displayed mode on the page
socket.on('mode_update', function(data) {
    document.getElementById('mode').textContent = data.mode;
});

// Handle mode change when the button is clicked
document.getElementById('changeModeButton').addEventListener('click', function() {
    // Check the current mode and switch to the other mode
    let newMode = document.getElementById('mode').textContent === 'Instruction' ? 'test' : 'instruction';
    
    // Emit a 'change_mode' event with the new mode to the server
    socket.emit('change_mode', newMode);
});
