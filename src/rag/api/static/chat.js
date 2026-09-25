const form = document.getElementById('chat-form');
const question = document.getElementById('question');
const messages = document.getElementById('messages');
const sendButton = document.getElementById('send-button');

function addMessage(text, kind) {
    const message = document.createElement('div');
    message.className = `message ${kind}`;
    message.textContent = text;
    messages.appendChild(message);
    message.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function askQuestion(text) {
    addMessage(text, 'user');
    sendButton.disabled = true;
    question.value = '';
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text, k: 4 }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || payload.error || 'Request failed');
        addMessage(payload.answer, 'assistant');
    } catch (error) {
        addMessage(`Unable to answer right now. ${error.message}`, 'assistant');
    } finally {
        sendButton.disabled = false;
        question.focus();
    }
}

form.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (text) askQuestion(text);
});

document.querySelectorAll('[data-prompt]').forEach((button) => {
    button.addEventListener('click', () => {
        question.value = button.dataset.prompt;
        question.focus();
    });
});

question.addEventListener('input', () => {
    question.style.height = 'auto';
    question.style.height = `${Math.min(question.scrollHeight, 130)}px`;
});
