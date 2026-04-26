import { useState } from 'react';
import { Send, CheckCircle2, ChevronLeft, Upload, X, AlertTriangle } from 'lucide-react';
import { ticketService } from '../services/apiService';

export default function CustomerPortal() {
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [fraudDetected, setFraudDetected] = useState(false);
  const [formData, setFormData] = useState({
    subject: '',
    message: '',
    customer_name: 'John Doe',
    customer_email: 'john@example.com',
    channel: 'web'
  });

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setFraudDetected(false);
    
    try {
      console.log('Submitting ticket...', { hasImage: !!imageFile });
      let result;
      if (imageFile) {
        console.log('Uploading with image...');
        result = await ticketService.createTicketWithImage(formData, imageFile);
      } else {
        console.log('Uploading without image...');
        result = await ticketService.createTicket({
          ...formData,
          hasImage: false
        });
      }
      
      console.log('Result:', result);
      
      if (result.fraud_detected) {
        setFraudDetected(true);
      }
      
      setSubmitted(true);
    } catch (err) {
      console.error('Submission error:', err);
      alert('Failed to submit ticket: ' + err.message + '\n\nPlease check:\n1. Backend is running on port 8000\n2. Browser console for errors (F12)');
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#f8fafc', color: '#0f172a', fontFamily: 'Inter' }}>
        {fraudDetected ? (
          <>
            <AlertTriangle size={64} color="#ef4444" style={{ marginBottom: 24 }} />
            <h1 style={{ fontSize: '2rem', marginBottom: 12, color: '#ef4444' }}>Fraud Alert Detected</h1>
            <p style={{ color: '#64748b', marginBottom: 32, textAlign: 'center', maxWidth: 500 }}>
              Our AI system has detected potential image manipulation in your submission. 
              Your ticket has been escalated to our fraud investigation team for manual review.
            </p>
          </>
        ) : (
          <>
            <CheckCircle2 size={64} color="#10b981" style={{ marginBottom: 24 }} />
            <h1 style={{ fontSize: '2rem', marginBottom: 12 }}>Request Received</h1>
            <p style={{ color: '#64748b', marginBottom: 32 }}>Our AI support team is reviewing your ticket.</p>
          </>
        )}
        <button 
          onClick={() => {
            setSubmitted(false);
            setFraudDetected(false);
            setImageFile(null);
            setImagePreview(null);
          }}
          style={{ background: '#3b82f6', color: 'white', padding: '12px 24px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 500 }}
        >
          Submit Another Request
        </button>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', padding: 40, fontFamily: 'Inter' }}>
      
      <div style={{ maxWidth: 600, margin: '0 auto', background: 'white', borderRadius: 16, padding: 40, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
        
        {/* Mock Client Branding */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 40, paddingBottom: 20, borderBottom: '1px solid #e2e8f0' }}>
          <div style={{ width: 40, height: 40, borderRadius: 8, background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 'bold', fontSize: 20 }}>
            F
          </div>
          <div>
            <h2 style={{ margin: 0, color: '#0f172a', fontSize: '1.2rem' }}>FoodDelivery Pro</h2>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Customer Support</div>
          </div>
        </div>

        <h1 style={{ color: '#0f172a', fontSize: '1.8rem', marginBottom: 8 }}>Report an Issue</h1>
        <p style={{ color: '#64748b', marginBottom: 32, fontSize: '0.95rem' }}>
          Submit a request with photo evidence. Our AI will analyze it for authenticity.
        </p>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          <div style={{ display: 'flex', gap: 12 }}>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: 8 }}>Full Name</label>
              <input 
                required type="text" placeholder="John Doe"
                value={formData.customer_name}
                onChange={(e) => setFormData({...formData, customer_name: e.target.value})}
                style={{ width: '100%', padding: '12px', borderRadius: 8, border: '1px solid #cbd5e1', outline: 'none', boxSizing: 'border-box' }}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: 8 }}>Email</label>
              <input 
                required type="email" placeholder="john@example.com"
                value={formData.customer_email}
                onChange={(e) => setFormData({...formData, customer_email: e.target.value})}
                style={{ width: '100%', padding: '12px', borderRadius: 8, border: '1px solid #cbd5e1', outline: 'none', boxSizing: 'border-box' }}
              />
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: 8 }}>Channel Origin (For testing):</label>
            <select 
              value={formData.channel}
              onChange={(e) => setFormData({...formData, channel: e.target.value})}
              style={{ width: '100%', padding: '12px', borderRadius: 8, border: '1px solid #cbd5e1', outline: 'none', background: '#f8fafc', color: '#0f172a' }}
            >
              <option value="web">Web Form</option>
              <option value="email">Email</option>
              <option value="social">Social Media (Twitter)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: 8 }}>Issue Subject</label>
            <input 
              required
              type="text" 
              placeholder="E.g., Found insect in my food"
              value={formData.subject}
              onChange={(e) => setFormData({...formData, subject: e.target.value})}
              style={{ width: '100%', padding: '12px', borderRadius: 8, border: '1px solid #cbd5e1', outline: 'none', boxSizing: 'border-box' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: 8 }}>Detailed Description</label>
            <textarea 
              required
              rows={5}
              placeholder="Describe the issue. Upload a photo for faster processing."
              value={formData.message}
              onChange={(e) => setFormData({...formData, message: e.target.value})}
              style={{ width: '100%', padding: '12px', borderRadius: 8, border: '1px solid #cbd5e1', outline: 'none', resize: 'vertical', boxSizing: 'border-box' }}
            />
          </div>

          {/* Image Upload Section */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: 8 }}>
              Upload Photo Evidence (Optional)
            </label>
            <div style={{ 
              border: '2px dashed #cbd5e1', 
              borderRadius: 8, 
              padding: 20, 
              textAlign: 'center',
              background: '#f8fafc',
              position: 'relative'
            }}>
              {imagePreview ? (
                <div style={{ position: 'relative' }}>
                  <img src={imagePreview} alt="Preview" style={{ maxWidth: '100%', maxHeight: 300, borderRadius: 8 }} />
                  <button
                    type="button"
                    onClick={removeImage}
                    style={{
                      position: 'absolute',
                      top: 10,
                      right: 10,
                      background: '#ef4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '50%',
                      width: 32,
                      height: 32,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer'
                    }}
                  >
                    <X size={18} />
                  </button>
                </div>
              ) : (
                <>
                  <Upload size={32} color="#94a3b8" style={{ marginBottom: 12 }} />
                  <p style={{ color: '#64748b', fontSize: '0.85rem', marginBottom: 12 }}>
                    Click to upload or drag and drop
                  </p>
                  <p style={{ color: '#94a3b8', fontSize: '0.75rem' }}>
                    PNG, JPG up to 10MB
                  </p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      opacity: 0,
                      cursor: 'pointer'
                    }}
                  />
                </>
              )}
            </div>
            <p style={{ fontSize: '0.72rem', color: '#f59e0b', marginTop: 8 }}>
              ⚠️ Our AI will analyze images for authenticity and detect manipulation
            </p>
          </div>

          <button 
            type="submit" 
            disabled={loading}
            style={{ 
              background: '#0f172a', color: 'white', padding: '14px', borderRadius: 8, border: 'none', 
              fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, marginTop: 12 
            }}
          >
            {loading ? 'Processing via AI Fraud Detection...' : 'Submit Ticket'} <Send size={16} />
          </button>
        </form>

      </div>
    </div>
  );
}
