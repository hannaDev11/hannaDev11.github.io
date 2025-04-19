function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const formObject = Object.fromEntries(formData.entries());

    // Show loading screen and hide form
    document.getElementById("loadingScreen").classList.remove("hidden");
    document.getElementById("predictionForm").classList.add("hidden");
    document.getElementById("resultPage").classList.add("hidden");

    fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify(formObject),
    })

    .then((response) => response.json())
    .then((data) => {
        document.getElementById("loadingScreen").classList.add("hidden");
        document.getElementById("resultPage").classList.remove("hidden");

        if (data.prediction) {
            document.getElementById("predictionResult").textContent = data.prediction;
            document.getElementById("errorResult").textContent = "";
        } else if (data.error) {
            document.getElementById("errorResult").textContent = data.error;
            document.getElementById("predictionResult").textContent = "";
        }
    })
    .catch((error) => {
        document.getElementById("loadingScreen").classList.add("hidden");
        document.getElementById("errorResult").textContent = "An error occurred while processing your request.";
        console.error("Error:", error);
    });
}

function resetForm() {
    document.getElementById("predictionForm").reset();
    document.getElementById("predictionForm").classList.remove("hidden");
    document.getElementById("resultPage").classList.add("hidden");
}
