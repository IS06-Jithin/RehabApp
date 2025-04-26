import { useEffect, useState } from "react";
import "./Feedback.css";          // <—  new file just for feedback styles

const API = "http://localhost:8000";

export default function FeedbackPage() {
  const [form, setForm] = useState({
    ease_of_use : 5,
    accuracy    : 5,
    satisfaction: 5,
    comments    : "",
  });
  const [avg, setAvg] = useState(null);
  const [msg, setMsg] = useState("");

  /* fetch averages once */
  useEffect(() => {
    fetch(`${API}/feedback/summary`).then(r => r.json()).then(setAvg);
  }, []);

  const handleChange = e =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const submit = async e => {
    e.preventDefault();
    const res = await fetch(`${API}/feedback`, {
      method : "POST",
      headers: { "Content-Type":"application/json" },
      body   : JSON.stringify({
        ...form,
        ease_of_use : +form.ease_of_use,
        accuracy    : +form.accuracy,
        satisfaction: +form.satisfaction,
      }),
    });
    if (res.ok) {
      setMsg("✅  Thanks for your feedback!");
      const summary = await fetch(`${API}/feedback/summary`).then(r => r.json());
      setAvg(summary);
    } else {
      setMsg("❌  Something went wrong.");
    }
  };

  /* ─────────────────────────── UI ─────────────────────────── */
  return (
    <div className="fb-wrapper">
      <h2>Provide Your Feedbacks</h2>

      {avg && (
        <p className="avg-line">
          Current average ratings (1-5):&nbsp;
          Ease&nbsp;{avg.avg_ease},&nbsp;
          Accuracy&nbsp;{avg.avg_accuracy},&nbsp;
          Satisfaction&nbsp;{avg.avg_satisf}
          &nbsp;({avg.count}&nbsp;responses)
        </p>
      )}

      <form onSubmit={submit} className="fb-form">
        {["ease_of_use", "accuracy", "satisfaction"].map(k => (
          <div key={k} className="fb-field">
            <label>{k.replace("_"," ").replace(/\b\w/g,l=>l.toUpperCase())}</label>
            <input
              type="range" min="1" max="5" step="1"
              name={k} value={form[k]} onChange={handleChange}
            />
            <span className="slider-val">{form[k]}</span>
          </div>
        ))}

        <div className="fb-field fb-comments">
          <label>Comments (optional)</label>
          <textarea
            name="comments" rows="3"
            value={form.comments}
            onChange={handleChange}
          />
        </div>

        <button className="btn btn-start" type="submit">Submit</button>
      </form>

      {msg && <p className="msg">{msg}</p>}

      <button
        className="btn btn-secondary back-btn"
        onClick={() => window.location.href = "/"}
      >
        ← Back to Therapy
      </button>
    </div>
  );
}
