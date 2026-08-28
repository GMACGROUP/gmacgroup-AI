document.addEventListener('DOMContentLoaded', () => { 
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const chatMessages = document.getElementById('chat-messages');
    const clearBtn = document.getElementById('clear-btn');
    const suggestions = document.querySelectorAll('.suggestion-chip');

    // State
    let chatHistory = [
        ["Hi", "Hey! I’m Chrix Tech. What would you like to ask—projects, skills, or availability?"]
    ];
    let isWaitingForResponse = false;
    let abortController = new AbortController();

    // Enable/disable send button based on input
    messageInput.addEventListener('input', () => {
        sendBtn.disabled = messageInput.value.trim() === '' || isWaitingForResponse;
    });

    // Handle suggestion clicks
    suggestions.forEach(chip => {
        chip.addEventListener('click', () => {
            const text = chip.textContent;
            messageInput.value = text;
            sendBtn.disabled = false;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    clearBtn.addEventListener('click', () => {
        chatHistory = [
            ["Hi", "Hey! I’m Chrix Tech. What would you like to ask—projects, skills, or availability?"]
        ];
        chatMessages.innerHTML = `
            <div class="message ai-message">
                <div class="message-content">Hey! I’m Chrix Tech. What would you like to ask—projects, skills, or availability?</div>
                <span class="message-time" aria-hidden="true">Just now</span>
            </div>
        `;
    });

    // Form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const messageText = messageInput.value.trim();
        if (!messageText || isWaitingForResponse) return;

        // Add user message to UI
        addMessageToUI('user', messageText);
        
        // Clear input and state
        messageInput.value = '';
        sendBtn.disabled = true;
        isWaitingForResponse = true;
        
        // Toggle buttons
        sendBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            // Reset abort controller
            abortController = new AbortController();

            // Send to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: messageText,
                    history: chatHistory
                }),
                signal: abortController.signal
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            const aiReply = data.reply;
            const newSuggestions = data.suggestions || [];

            // Remove typing indicator
            removeTypingIndicator(typingId);

            // Add AI response to UI
            addMessageToUI('ai', aiReply);

            // Update history
            chatHistory.push([messageText, aiReply]);

            // Update suggestions UI
            updateSuggestions(newSuggestions);

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Fetch aborted by user.');
                removeTypingIndicator(typingId);
                addMessageToUI('ai', 'Generation stopped.');
            } else {
                console.error('Error:', error);
                removeTypingIndicator(typingId);
                addMessageToUI('ai', 'Oops, something went wrong connecting to the server. Please try again.');
            }
        } finally {
            isWaitingForResponse = false;
            
            // Toggle buttons back
            stopBtn.classList.add('hidden');
            sendBtn.classList.remove('hidden');

            // Re-enable send button if there's text
            sendBtn.disabled = messageInput.value.trim() === '';

            // Focus input if not on mobile
            if (window.innerWidth > 768) {
                messageInput.focus();
            }
        }
    });

    // Stop generation
    stopBtn.addEventListener('click', () => {
        if (isWaitingForResponse) {
            abortController.abort();
        }
    });

    // Helpers
    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }

    function urlify(text) {
        const escaped = escapeHTML(text);
        // Match URLs but exclude trailing punctuation like periods or commas
        const urlRegex = /(https?:\/\/[^\s]+[^\s.,;:!?'"()])/g;
        return escaped.replace(urlRegex, function(url) {
            return `<a href="${url}" target="_blank" rel="noopener noreferrer" style="color: #58a6ff; text-decoration: underline;">${url}</a>`;
        });
    }

    function getTimeLabel() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function addMessageToUI(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = urlify(text);

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.setAttribute('aria-hidden', 'true');
        timeSpan.textContent = getTimeLabel();

        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeSpan);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message';
        typingDiv.id = id;
        
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="spinner"></div>
            </div>
        `;
        
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Handle mobile keyboard appearance cleanly
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', () => {
            scrollToBottom();
        });
        window.visualViewport.addEventListener('scroll', () => {
            // Keep window anchored at top
            if (window.scrollY !== 0) {
                window.scrollTo(0, 0);
            }
        });
    }

    // Scroll to bottom when focusing the input on mobile
    messageInput.addEventListener('focus', () => {
        setTimeout(scrollToBottom, 250);
    });

    function updateSuggestions(newSuggestions) {
        const suggestionsContainer = document.getElementById('suggestions');
        suggestionsContainer.innerHTML = ''; // Clear old suggestions
        
        newSuggestions.forEach(text => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.textContent = text;
            chip.addEventListener('click', () => {
                messageInput.value = text;
                sendBtn.disabled = false;
                chatForm.dispatchEvent(new Event('submit'));
            });
            suggestionsContainer.appendChild(chip);
        });
    }
});
