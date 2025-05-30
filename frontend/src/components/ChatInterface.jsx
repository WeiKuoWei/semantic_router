import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkMath from 'remark-math';
import { InlineMath, BlockMath } from 'react-katex';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { sendQuery, getHistory, clearHistory } from '../services/api';

const ThinkingAnimation = () => {
  return (
    <div className="loading-indicator">
      Thinking
      <span className="dot">.</span>
      <span className="dot">.</span>
      <span className="dot">.</span>
    </div>
  );
};

const renderMath = ({ value, inline }) => {
  return inline ? <InlineMath math={value} /> : <BlockMath math={value} />;
};

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [loading, setLoading] = useState(false);
  
  const preprocessLatex = (text) => {
    // First, fix inline math expressions: $...$
    let processedText = text.replace(/\$([^$]+)\$/g, (match, formula) => {
      return '$' + formula.replace(/\\([a-zA-Z]+)/g, '\\$1') + '$';
    });
    
    // Then, fix block math expressions: $$...$$
    processedText = processedText.replace(/\$\$([^$]+)\$\$/g, (match, formula) => {
      return '$$' + formula.replace(/\\([a-zA-Z]+)/g, '\\$1') + '$$';
    });
    
    // Also fix explicit math blocks with brackets: [ ... ]
    processedText = processedText.replace(/\[(.*?)\]/g, (match, formula) => {
      if (formula.includes('\\leftarrow') || formula.includes('\\alpha') || 
          formula.includes('\\gamma') || formula.includes('\\max_')) {
        return '[' + formula.replace(/\\([a-zA-Z_]+)/g, '\\$1') + ']';
      }
      return match; // Not a math formula, return unchanged
    });
    
    return processedText;
  };

  useEffect(() => {
    // Initialize session ID from localStorage or create new one
    const storedSessionId = localStorage.getItem('sessionId');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      // Load history for this session
      loadHistory(storedSessionId);
    } else {
      const newSessionId = `user_${Date.now()}`;
      setSessionId(newSessionId);
      localStorage.setItem('sessionId', newSessionId);
    }
  }, []);

  const loadHistory = async (sid) => {
    try {
      const response = await getHistory(sid);
      if (response.history) {
        const formattedMessages = response.history.map(msg => ({
          id: `user-${msg.timestamp}`,
          text: msg.query,
          sender: 'user',
          timestamp: msg.timestamp
        })).concat(response.history.map(msg => ({
          id: `bot-${msg.timestamp + 1}`,
          text: msg.response,
          sender: 'bot',
          expert: msg.expert,
          timestamp: msg.timestamp + 1 // Just to ensure order
        }))).sort((a, b) => a.timestamp - b.timestamp);
        
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const timestamp = Date.now();
    // Add user message to UI
    const userMessage = {
      id: `user-${timestamp}`,
      text: input,
      sender: 'user',
      timestamp
    };
    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Send to backend
      const response = await sendQuery(input, sessionId);
      
      const responseTimestamp = Date.now();
      // Add response to UI
      const botMessage = {
        id: `bot-${responseTimestamp}`,
        text: response.answer,
        sender: 'bot',
        expert: response.expert,
        sources: response.sources,
        timestamp: responseTimestamp
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages(prev => [...prev, {
        id: `error-${Date.now()}`,
        text: 'Sorry, there was an error processing your request.',
        sender: 'bot',
        error: true,
        timestamp: Date.now()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await clearHistory(sessionId);
      setMessages([]);
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };

  // Components for markdown rendering
  const components = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={atomDark}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
    // Add math rendering support
    math: renderMath,
    inlineMath: ({ node, ...props }) => renderMath({ ...props, inline: true }),
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Expert AI Assistant</h2>
        <button onClick={handleReset} className="clear-btn">Clear Chat</button>
      </div>
      
      <div className="messages-container">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender} ${msg.error ? 'error' : ''}`}>
            <div className="message-content">
              {msg.sender === 'bot' ? (
                <ReactMarkdown 
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={components}
                >
                  {preprocessLatex(msg.text)}
                </ReactMarkdown>
              ) : (
                msg.text
              )}
            </div>
            {msg.expert && (
              <div className="message-expert">Expert: {msg.expert}</div>
            )}
            {msg.sources && (
              <div className="message-sources">Sources: {msg.sources}</div>
            )}
          </div>
        ))}
        {loading && <ThinkingAnimation />}
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
          className="message-input"
        />
        <button type="submit" disabled={loading || !input.trim()} className="send-btn">
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;