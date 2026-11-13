
import { useState, useRef } from "react";
import {
  Bot,
  Send,
  FileText,
  Upload,
  Sparkles,
  Trash2,
  LoaderCircle,
  CheckCircle2,
  AlertCircle,
  MessageSquare,
  BookOpen,
  ShieldCheck,
  X,
} from "lucide-react";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [error, setError] = useState("");

  const fileInputRef = useRef(null);

  // Select PDF
  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (
      selectedFile.type !== "application/pdf" &&
      !selectedFile.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("Please select a valid PDF file.");
      return;
    }

    if (selectedFile.size > 25 * 1024 * 1024) {
      setError("PDF size must be less than 25 MB.");
      return;
    }

    setError("");
    setFile(selectedFile);
  };

  // Upload PDF
  const uploadPDF = async () => {
    if (!file) {
      setError("Please select a PDF first.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "PDF upload failed."
        );
      }

      setUploadedFile({
        name: file.name,
        details: data,
      });

      setFile(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

    } catch (err) {
      setError(
        err.message ||
        "Unable to upload PDF. Check the backend."
      );
    } finally {
      setUploading(false);
    }
  };

  // Ask AI
  const askQuestion = async () => {
    const currentQuestion = question.trim();

    if (!currentQuestion || loading) return;

    setError("");

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: currentQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/chat/ask`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            question: currentQuestion,
            top_k: 5,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to generate answer."
        );
      }

      const result = data.data;

      if (!result || !result.answer) {
        throw new Error("Invalid response received from AI.");
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources || [],
          executionTime: result.execution_time,
          retrievedChunks: result.retrieved_chunks,
        },
      ]);

    } catch (err) {
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: `Error: ${err.message}`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Clear conversation
  const clearChat = () => {
    setMessages([]);
    setError("");
  };

  // Format sources
  const renderSources = (sources) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="sources-container">
        <h4>
          <BookOpen size={15} />
          Sources
        </h4>

        {sources.map((source, index) => {
          const filename =
            typeof source === "string"
              ? source
              : source.filename || "Document";

          const page =
            typeof source === "object"
              ? source.page_number
              : null;

          return (
            <div className="source-item" key={index}>
              <FileText size={15} />

              <span>{filename}</span>

              {page && (
                <span className="source-page">
                  Page {page}
                </span>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="app-container">

      {/* NAVBAR */}

      <header className="navbar">

        <div className="brand">
          <div className="brand-icon">
            <Bot size={24} />
          </div>

          <div>
            <h1>DocuMind AI</h1>
            <span>Advanced RAG Intelligence</span>
          </div>
        </div>

        <div className="navbar-status">
          <span className="status-dot"></span>
          AI Workspace
        </div>

      </header>

      {/* MAIN */}

      <main className="main-content">

        {/* HERO */}

        <section className="hero">

          <div className="hero-badge">
            <Sparkles size={14} />
            Intelligent Document Assistant
          </div>

          <h2>
            Your Documents.
            <span> Your AI Assistant.</span>
          </h2>

          <p>
            Upload your documents, ask complex questions,
            and discover information with context-aware AI.
          </p>

          <div className="hero-features">

            <span>
              <ShieldCheck size={16} />
              Document Grounded
            </span>

            <span>
              <Sparkles size={16} />
              AI Powered
            </span>

            <span>
              <BookOpen size={16} />
              Source Citations
            </span>

          </div>

        </section>

        {/* WORKSPACE */}

        <div className="workspace">

          {/* UPLOAD PANEL */}

          <section className="panel upload-card">

            <div className="panel-heading">

              <div className="panel-icon">
                <Upload size={21} />
              </div>

              <div>
                <h3>Upload Documents</h3>
                <p>Add PDF files to your knowledge base.</p>
              </div>

            </div>

            <div
              className="upload-area"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();

                const droppedFile = event.dataTransfer.files?.[0];

                if (droppedFile) {
                  handleFileChange({
                    target: {
                      files: [droppedFile],
                    },
                  });
                }
              }}
            >

              <div className="upload-icon">
                <Upload size={27} />
              </div>

              <h4>Drop your PDF here</h4>

              <p>or click to browse files</p>

              <span>PDF files · Maximum 25 MB</span>

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
                hidden
              />

            </div>

            {file && (

              <div className="selected-file">

                <FileText size={19} />

                <div className="selected-file-info">
                  <strong>{file.name}</strong>
                  <span>
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>

                <button
                  className="icon-button"
                  onClick={() => setFile(null)}
                  disabled={uploading}
                  aria-label="Remove selected file"
                >
                  <X size={17} />
                </button>

              </div>

            )}

            <button
              className="primary-button upload-button"
              onClick={uploadPDF}
              disabled={!file || uploading}
            >

              {uploading ? (
                <>
                  <LoaderCircle className="spin" size={18} />
                  Processing...
                </>
              ) : (
                <>
                  <Upload size={18} />
                  Upload Document
                </>
              )}

            </button>

            {uploadedFile && (

              <div className="upload-success">
                <CheckCircle2 size={17} />

                <span>
                  Uploaded: {uploadedFile.name}
                </span>

              </div>

            )}

            <div className="privacy-note">
              <ShieldCheck size={15} />
              Your document is processed by your local AI backend.
            </div>

          </section>

          {/* CHAT PANEL */}

          <section className="panel chat-card">

            <div className="chat-header">

              <div className="panel-heading">

                <div className="panel-icon">
                  <MessageSquare size={21} />
                </div>

                <div>
                  <h3>Ask DocuMind</h3>
                  <p>Chat with your document knowledge base.</p>
                </div>

              </div>

              <button
                className="clear-button"
                onClick={clearChat}
                disabled={messages.length === 0 || loading}
              >
                <Trash2 size={16} />
                Clear
              </button>

            </div>

            {/* MESSAGES */}

            <div className="chat-messages">

              {messages.length === 0 && (

                <div className="empty-chat">

                  <div className="empty-chat-icon">
                    <Bot size={34} />
                  </div>

                  <h3>How can I help you?</h3>

                  <p>
                    Ask questions about your uploaded documents
                    and get intelligent, contextual answers.
                  </p>

                  <div className="suggestions">

                    <button
                      onClick={() =>
                        setQuestion("Summarize the document.")
                      }
                    >
                      Summarize document
                    </button>

                    <button
                      onClick={() =>
                        setQuestion(
                          "Explain the important concepts."
                        )
                      }
                    >
                      Explain key concepts
                    </button>

                  </div>

                </div>

              )}

              {messages.map((message, index) => (

                <div
                  key={index}
                  className={`message ${message.role}`}
                >

                  <div className="message-avatar">

                    {message.role === "user" ? (
                      "You"
                    ) : (
                      <Bot size={18} />
                    )}

                  </div>

                  <div className="message-content">

                    <strong>
                      {message.role === "user"
                        ? "You"
                        : "DocuMind AI"}
                    </strong>

                    <p
                      className={
                        message.isError ? "error-text" : ""
                      }
                    >
                      {message.content}
                    </p>

                    {message.role === "assistant" &&
                      !message.isError && (
                        <>
                          {renderSources(message.sources)}

                          {message.executionTime !== undefined && (
                            <div className="answer-meta">
                              {message.retrievedChunks ?? 0} chunks
                              {" · "}
                              {message.executionTime}s
                            </div>
                          )}
                        </>
                      )}

                  </div>

                </div>

              ))}

              {loading && (

                <div className="thinking-indicator">

                  <LoaderCircle className="spin" size={18} />

                  DocuMind is thinking...

                </div>

              )}

            </div>

            {/* INPUT */}

            <div className="chat-input">

              <input
                type="text"
                placeholder="Ask something about your documents..."
                value={question}
                onChange={(event) =>
                  setQuestion(event.target.value)
                }
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !loading
                  ) {
                    event.preventDefault();
                    askQuestion();
                  }
                }}
                disabled={loading}
              />

              <button
                className="send-button"
                onClick={askQuestion}
                disabled={!question.trim() || loading}
                aria-label="Send message"
              >

                {loading ? (
                  <LoaderCircle className="spin" size={19} />
                ) : (
                  <Send size={19} />
                )}

              </button>

            </div>

            <p className="chat-disclaimer">
              DocuMind AI generates answers using your stored documents.
            </p>

          </section>

        </div>

        {/* FOOTER */}

        <footer className="footer">

          <div className="footer-brand">
            <Bot size={17} />
            DocuMind AI
          </div>

          <span>
            Advanced Retrieval-Augmented Generation
          </span>

        </footer>

      </main>

      {error && (

        <div className="error-toast">

          <AlertCircle size={18} />

          {error}

          <button onClick={() => setError("")}>
            <X size={16} />
          </button>

        </div>

      )}

    </div>
  );
}

export default App;
