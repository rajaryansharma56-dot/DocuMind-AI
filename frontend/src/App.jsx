
import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadPDF = async () => {
    if (!file) {
      alert("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      alert("PDF uploaded successfully!");
      console.log(data);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    const currentQuestion = question;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: currentQuestion },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: currentQuestion }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate answer");
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || data.response || JSON.stringify(data),
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error: " + error.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="navbar">
        <h1>DocuMind AI</h1>
        <span>Advanced RAG Intelligence</span>
      </header>

      <main className="main-content">
        <section className="hero">
          <h2>Your Documents. Your AI Assistant.</h2>
          <p>
            Upload documents, ask questions, and get intelligent,
            context-aware answers powered by AI.
          </p>
        </section>

        <section className="upload-card">
          <h2>Upload Your PDF</h2>

          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
          />

          {file && <p>Selected: {file.name}</p>}

          <button onClick={uploadPDF} disabled={loading}>
            {loading ? "Processing..." : "Upload Document"}
          </button>
        </section>

        <section className="chat-card">
          <h2>Ask DocuMind</h2>

          <div className="chat-messages">
            {messages.length === 0 && (
              <p className="empty-message">
                Ask anything about your uploaded documents.
              </p>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.role}`}
              >
                <strong>
                  {message.role === "user" ? "You" : "DocuMind AI"}
                </strong>
                <p>{message.content}</p>
              </div>
            ))}
          </div>

          <div className="chat-input">
            <input
              type="text"
              placeholder="Ask a question about your PDF..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !loading) askQuestion();
              }}
            />

            <button onClick={askQuestion} disabled={loading}>
              {loading ? "Thinking..." : "Ask"}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
