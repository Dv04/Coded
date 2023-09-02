import ChatGPT from './chatgpt';

// Create a new ChatGPT instance
const chatgpt = new ChatGPT();

// Add a listener for the `search` event
chatgpt.on('search', async (query) => {
    // Search for the query using Google Search
    const results = await google.search(query);

    // Display the results in a div
    const div = document.getElementById('results');
    div.innerHTML = results;
});
