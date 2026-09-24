/* ============================================================
    GMAC GROUP COMPANY ASSISTANT — CLIENT APPLICATION SCRIPT
    Interactive Company Assistant, Clean Layout, Copy Action Toolbar
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const chatMessages = document.getElementById('chat-messages');
    const clearBtn = document.getElementById('clear-btn');
    const aboutBtn = document.getElementById('about-btn');
    const headerProfile = document.getElementById('header-profile-info');
    const bioModal = document.getElementById('bio-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const suggestionsContainer = document.getElementById('suggestions');
    const scrollBottomBtn = document.getElementById('scroll-bottom-btn');
    const toast = document.getElementById('toast');
    const toastText = document.getElementById('toast-text');

    // Opening greetings pool
    const OPENING_GREETINGS = [
        "Welcome to Gmac Group. Ask about our research, human capital, events, investment facilitation, or how to begin a conversation.",
        "Welcome to Gmac Group. I can explain our practice areas, engagement models, team, events, and contact channels.",
        "Gmac Group connects talent to opportunity through research, human capital, and investment facilitation. What would you like to explore?"
    ];

    const chosenGreeting = OPENING_GREETINGS[Math.floor(Math.random() * OPENING_GREETINGS.length)];

    // State
    let chatHistory = [
        ["Hi", chosenGreeting]
    ];
    let isWaitingForResponse = false;
    let abortController = new AbortController();
    let toastTimeout = null;

    // Avatar path
    const AVATAR_URL = document.querySelector('.avatar-img')?.src || '/static/avatar.jpg';

    // ── Helper: Format Time ──
    function getTimeLabel() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    // ── Helper: Escape HTML ──
    function escapeHTML(str) {
        return (str || '').replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    }

    // ── Rich Markdown Parser ──
    function renderMarkdown(rawText) {
        if (!rawText) return '';

        // Extract code blocks first
        const codeBlocks = [];
        let text = rawText.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const index = codeBlocks.length;
            codeBlocks.push({ lang: lang || 'code', code: code.trim() });
            return `__CODE_BLOCK_${index}__`;
        });

        text = escapeHTML(text);

        // Inline code `code`
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold **text** or __text__
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');

        // Italic *text* or _text_
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        text = text.replace(/_([^_]+)_/g, '<em>$1</em>');

        // Autolink URLs
        const urlRegex = /(https?:\/\/[^\s<]+[^\s.,;:!?'"()<>])/g;
        text = text.replace(urlRegex, (url) => {
            return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
        });

        // Lists & paragraphs
        const lines = text.split('\n');
        let inList = false;
        let formattedLines = [];

        for (let line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ')) {
                if (!inList) {
                    formattedLines.push('<ul>');
                    inList = true;
                }
                formattedLines.push(`<li>${trimmed.substring(2)}</li>`);
            } else {
                if (inList) {
                    formattedLines.push('</ul>');
                    inList = false;
                }
                if (trimmed.length > 0) {
                    formattedLines.push(`<p>${line}</p>`);
                }
            }
        }
        if (inList) formattedLines.push('</ul>');
        let htmlResult = formattedLines.join('');

        // Re-inject code blocks
        htmlResult = htmlResult.replace(/__CODE_BLOCK_(\d+)__/g, (match, index) => {
            const item = codeBlocks[parseInt(index, 10)];
            const safeCode = escapeHTML(item.code);
            return `
                <div class="code-block-wrapper">
                    <div class="code-block-header">
                        <span>${escapeHTML(item.lang)}</span>
                        <button class="copy-code-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('${encodeURIComponent(item.code)}')); this.innerText='Copied!'; setTimeout(()=>this.innerText='Copy', 1800);">Copy</button>
                    </div>
                    <pre><code>${safeCode}</code></pre>
                </div>
            `;
        });

        return htmlResult;
    }

    // ── Toast Notification ──
    function showToast(message) {
        if (toastTimeout) clearTimeout(toastTimeout);
        toastText.textContent = message;
        toast.classList.remove('hidden');
        toastTimeout = setTimeout(() => {
            toast.classList.add('hidden');
        }, 2200);
    }

    // ── Scroll to Bottom ──
    function scrollToBottom(smooth = true) {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: smooth ? 'smooth' : 'auto'
        });
    }

    chatMessages.addEventListener('scroll', () => {
        const distanceFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
        if (distanceFromBottom > 140) {
            scrollBottomBtn.classList.remove('hidden');
        } else {
            scrollBottomBtn.classList.add('hidden');
        }
    });

    scrollBottomBtn.addEventListener('click', () => {
        scrollToBottom(true);
    });

    // ── Add Message To UI ──
    function addMessageToUI(sender, text) {
        const row = document.createElement('div');
        row.className = `message-row ${sender}-row`;

        const timeStr = getTimeLabel();

        if (sender === 'ai') {
            row.innerHTML = `
                <div class="msg-avatar-col">
                    <img src="${AVATAR_URL}" alt="Gmac Group" class="msg-avatar-img">
                </div>
                <div class="message-bubble-wrap">
                    <div class="msg-header-info">
                        <span class="msg-sender-name">Gmac Group</span>
                        <span class="msg-ai-pill">AI</span>
                        <span class="msg-time">${timeStr}</span>
                    </div>
                    <div class="message-bubble">
                        ${renderMarkdown(text)}
                    </div>
                    <div class="msg-actions-toolbar">
                        <button class="msg-action-btn copy-msg-btn" title="Copy response to clipboard">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy</span>
                        </button>
                    </div>
                </div>
            `;

            const copyBtn = row.querySelector('.copy-msg-btn');
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(text).then(() => {
                    showToast("Response copied to clipboard");
                    copyBtn.innerHTML = `
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                        <span style="color:#10b981">Copied!</span>
                    `;
                    setTimeout(() => {
                        copyBtn.innerHTML = `
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy</span>
                        `;
                    }, 1800);
                });
            });

        } else {
            row.innerHTML = `
                <div class="message-bubble-wrap">
                    <div class="msg-header-info">
                        <span class="msg-time">${timeStr}</span>
                        <span class="msg-sender-name">You</span>
                    </div>
                    <div class="message-bubble">
                        ${renderMarkdown(text)}
                    </div>
                </div>
            `;
        }

        chatMessages.appendChild(row);
        scrollToBottom(true);
    }

    // Initialize with the chosen opening greeting
    addMessageToUI('ai', chosenGreeting);

    // ── Suggestion Chips Handler ──
    function bindSuggestionChips() {
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.onclick = () => {
                const text = chip.innerText.replace(/^[^\w\s]+/, '').trim();
                messageInput.value = text;
                sendBtn.disabled = false;
                chatForm.dispatchEvent(new Event('submit'));
            };
        });
    }
    bindSuggestionChips();

    // ── Input & Send Button State ──
    messageInput.addEventListener('input', () => {
        sendBtn.disabled = messageInput.value.trim() === '' || isWaitingForResponse;
    });

    // ── Clear Chat ──
    clearBtn.addEventListener('click', () => {
        const resetGreeting = OPENING_GREETINGS[Math.floor(Math.random() * OPENING_GREETINGS.length)];
        chatHistory = [["Hi", resetGreeting]];
        chatMessages.innerHTML = '';
        addMessageToUI('ai', resetGreeting);
        showToast("Conversation cleared");
    });

    // ── Bio Modal ──
    function openModal() { bioModal.classList.remove('hidden'); }
    function closeModal() { bioModal.classList.add('hidden'); }

    if (aboutBtn) aboutBtn.addEventListener('click', openModal);
    if (headerProfile) headerProfile.addEventListener('click', openModal);
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    if (bioModal) {
        bioModal.addEventListener('click', (e) => {
            if (e.target === bioModal) closeModal();
        });
    }
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !bioModal.classList.contains('hidden')) {
            closeModal();
        }
    });

    // ── Typing Indicator ──
    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const row = document.createElement('div');
        row.className = 'message-row ai-row';
        row.id = id;

        row.innerHTML = `
            <div class="msg-avatar-col">
                <img src="${AVATAR_URL}" alt="Gmac Group" class="msg-avatar-img">
            </div>
            <div class="message-bubble-wrap">
                <div class="typing-bubble">
                    <span>Thinking</span>
                    <div class="typing-dots">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            </div>
        `;

        chatMessages.appendChild(row);
        scrollToBottom(true);
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // ── Update Suggestions ──
    function updateSuggestions(newSuggestions) {
        if (!newSuggestions || newSuggestions.length === 0) return;

        const iconMap = ['⚡', '🧠', '🎓', '🚀', '💼', '💡'];
        suggestionsContainer.innerHTML = '';

        newSuggestions.forEach((text, i) => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.setAttribute('role', 'listitem');
            const icon = iconMap[i % iconMap.length];
            chip.innerHTML = `<span class="chip-icon">${icon}</span> ${escapeHTML(text)}`;
            suggestionsContainer.appendChild(chip);
        });

        bindSuggestionChips();
    }

    // ── Form Submission ──
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const messageText = messageInput.value.trim();
        if (!messageText || isWaitingForResponse) return;

        addMessageToUI('user', messageText);

        messageInput.value = '';
        sendBtn.disabled = true;
        isWaitingForResponse = true;

        sendBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');

        const typingId = showTypingIndicator();

        try {
            abortController = new AbortController();

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: messageText,
                    history: chatHistory
                }),
                signal: abortController.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const aiReply = data.reply || "I couldn't process that response.";
            const newSuggestions = data.suggestions || [];

            removeTypingIndicator(typingId);
            addMessageToUI('ai', aiReply);

            chatHistory.push([messageText, aiReply]);
            updateSuggestions(newSuggestions);

        } catch (error) {
            removeTypingIndicator(typingId);
            if (error.name === 'AbortError') {
                addMessageToUI('ai', '_Response generation stopped._');
            } else {
                console.error('Chat API Error:', error);
                addMessageToUI('ai', 'Oops, something went wrong connecting to the backend. Please try again.');
            }
        } finally {
            isWaitingForResponse = false;
            stopBtn.classList.add('hidden');
            sendBtn.classList.remove('hidden');
            sendBtn.disabled = messageInput.value.trim() === '';

            if (window.innerWidth > 768) {
                messageInput.focus();
            }
        }
    });

    // ── Stop Button ──
    stopBtn.addEventListener('click', () => {
        if (isWaitingForResponse) {
            abortController.abort();
        }
    });

    // ── Mobile Viewport ──
    if (window.visualViewport) {
        window.visualViewport.addEventListener('resize', () => {
            scrollToBottom(false);
        });
    }

    messageInput.addEventListener('focus', () => {
        setTimeout(() => scrollToBottom(false), 250);
    });
});
