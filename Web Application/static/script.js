function predict() {
    var text = document.getElementById('text').value.trim();
    
    if (text === '') {
        document.getElementById('prediction').innerHTML = 'Please Enter Some Text';
        return;
    }
    
    fetch('/suicide-ideation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text
        })
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        var predictionText = data.predictionText;
        document.getElementById('prediction').innerHTML = predictionText + ' (' + data.prediction.toFixed(2) + ')';
        
        const predictionDiv = document.getElementById('prediction');
        const animation = anime({
          targets: predictionDiv,
          translateY: ["-100%", 0],
          opacity: [0, 1],
          scale: [0.5, 1],
          duration: 1000,
          easing: "spring(1, 80, 10, 0)",
          delay: 500
        });
    })
    .catch(function(error) {
        document.getElementById('prediction').innerHTML = 'Error: ' + error.message;
    });
}
