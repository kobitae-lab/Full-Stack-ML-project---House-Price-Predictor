
async function predictPrice(){
    try{
        const MedInc = parseFloat(document.getElementById("MedInc").value);
        const HouseAge = parseFloat(document.getElementById("HouseAge").value);
        const AverageRooms = parseFloat(document.getElementById("AveRooms").value);
        const AveBedrooms = parseFloat(document.getElementById("AveBedrms").value);
        const Population = parseFloat(document.getElementById("Population").value);
        const AveOccup = parseFloat(document.getElementById("AveOccup").value);
        const Latitude = parseFloat(document.getElementById("Latitude").value);
        const Longitude = parseFloat(document.getElementById("Longitude").value);

        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: "POST",
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({"MedInc": MedInc, 
                                "HouseAge": HouseAge, 
                                "AveRooms": AverageRooms,
                                "AveBedrms": AveBedrooms,
                                "Population": Population,
                                "AveOccup": AveOccup,
                                "Latitude": Latitude,
                                "Longitude": Longitude
                                })
        })
    const data = await response.json();
    document.getElementById('Prediction').textContent = "Predicted Price: "; // reset
    document.getElementById('Prediction').textContent += "$" + data.Predicted_Price;

    }

    catch(error){
        console.log(error);
    }
}


async function loadHistory(){
    try {
        const response = await fetch('http://127.0.0.1:5000/predictions');
        const data = await response.json();
        const history = document.getElementById("history");
        history.innerHTML = '';

        for(let i = 0; i < data.predictions.length; i++){
            const newPara = document.createElement("p");
            newPara.textContent = "Predicted: " + data.predictions[i].predicted_price + ', ' + "Actual: " + data.predictions[i].actual_price;
            document.getElementById('history').appendChild(newPara);
        }
    }
    catch(error){
        console.log(error);
    }
}

window.onload = loadHistory;