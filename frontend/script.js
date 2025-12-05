const API_URL = "http://127.0.0.1:8000/predict";

document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const review = document.getElementById("reviewsInput").value.trim();

    if (!review) {
        alert("Please enter a review.");
        return;
    }

    const data = await analyze(review);

    let output = `
        <div class="review-card">
            <b>Review:</b> ${review}<br><br>

            <b>Sentiment:</b> 
            <span class="sentiment-${data.sentiment.toLowerCase()}">${data.sentiment}</span><br><br>

            <b>Meaning:</b> ${interpretMeaning(data.sentiment)}<br><br>

            <b>Cleaned Text:</b> ${data.cleaned_text}
        </div>
    `;

    document.getElementById("results").innerHTML = output;
});


async function analyze(review) {
    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review: review })
    });

    return response.json();
}


function interpretMeaning(sentiment) {
    if (sentiment === "Positive") {
        return "This sentence expresses appreciation, satisfaction, or a positive emotional tone.";
    } else {
        return "This sentence conveys dissatisfaction, frustration, or negative feelings about the experience.";
    }
}
