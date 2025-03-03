const signUpButton = document.getElementById('signUp');
const signInButton = document.getElementById('signIn');
const container = document.getElementById('container');

signUpButton.addEventListener('click', () => {
	container.classList.add("right-panel-active");
});

signInButton.addEventListener('click', () => {
	container.classList.remove("right-panel-active");
});

document.getElementById("skipBtn").addEventListener("click", function() {
    // Start the Flask app (if not running)
    fetch('/start-app').then(response => {
		console.log("fetching")
        if (response.ok) {
            // Redirect to chat.html after starting the app
			console.log("doing work")
			window.location.href = "file:///D:/Major Project/YT bot 2/Medical-Bot/templates/chat.html";
			console.log("work done")
        }
    });
});