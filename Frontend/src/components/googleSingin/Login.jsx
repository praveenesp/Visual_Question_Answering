import { auth, provider } from "../../config";
import { useState, useEffect } from "react";
import { Navigate } from 'react-router-dom';
import { signInWithPopup, signInWithEmailAndPassword } from "firebase/auth";
import Home from "../Home/Home";
import './Auth.css';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);

  const handleGoogleSignIn = () => {
    signInWithPopup(auth, provider)
      .then((result) => {
        setUser(result.user);
        localStorage.setItem("email", result.user.email);
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  const handleEmailSignIn = () => {
    signInWithEmailAndPassword(auth, email, password)
      .then((result) => {
        setUser(result.user);
        localStorage.setItem("email", result.user.email);
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  useEffect(() => {
    const email = localStorage.getItem("email");
    if (email) {
      setUser({ email });
      console.log({email});
    }
    
  }, []);

  return (
    <div>
      {user ? (
        
        <Navigate to="/home" />
 
      ) : (
      <><div className="nav">
            <span className="navTitle">👁️ VQA</span>
          </div><div className="auth-container">
              <h2>Sign In</h2>
              <div className="auth-form">
                <input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)} />
                <input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)} />
                <button className="auth-button" onClick={handleEmailSignIn}>Sign In with Email</button>
                <p className="or-text">------ OR ------</p>
                <button className="auth-button" onClick={handleGoogleSignIn}>Sign In with Google</button>
                {error && <p className="auth-error">{error}</p>}
              </div>
            </div></>
      )}
    </div>
  );
}
