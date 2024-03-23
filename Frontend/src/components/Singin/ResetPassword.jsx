import { useState } from 'react';
import {sendPasswordResetEmail } from 'firebase/auth';
import './ResetPassword.css';
import { auth } from "../../config";
const ResetPassword = () => {
  const [email, setEmail] = useState('');
  const [error, setError] = useState(null);
  const [resetSent, setResetSent] = useState(false);

  const handleResetPassword = () => {
    console.log(email);
    sendPasswordResetEmail(auth,email)
      .then(() => {
        setResetSent(true);
        alert("email sent");
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  return (
    <><div className="nav">
    <img src="https://res.cloudinary.com/dcfzq326i/image/upload/v1711189350/tvbbxznhzh1byutclut1.png" alt="Logo"></img>
          <span className="navTitle">PicProbe</span>
      </div>
      <div className="reset-password-container">
              <h2>Reset Password</h2>
              <div className="reset-form">
                  <input
                      type="email"
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)} />
                  <button onClick={handleResetPassword}>Reset Password</button>
                  {error && <p className="error-message">{error}</p>}
                  {resetSent && <p className="success-message">Password reset email sent. Check your inbox.</p>}
              </div>
          </div>
        </>
  );
};

export default ResetPassword;
