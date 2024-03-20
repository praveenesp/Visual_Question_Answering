import { useState } from "react";
import { Link } from "react-router-dom";// Assuming you're using React Router for navigation
import { auth} from "../../config";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import './Auth.css';

export default function Signup() {
  localStorage.clear();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState(null);

  const handleSignup = () => {
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        const user = userCredential.user;
        updateProfile(user, { displayName: username })
          .then(() => {
            alert("User profile updated ");
            
          })
          .catch((error) => {
            console.error("Error updating profile:", error);
          });
        console.log("User Name:",username);  
        console.log("User signed up:", user.email);
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  return (
    <><div className="nav">
      <span className="navTitle">👁️ VQA</span>
    </div><div className="auth-container">
        <h2>Sign Up</h2>
        <div className="auth-form">
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)} />
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
          <button className="auth-button" onClick={handleSignup}>Sign Up</button>
          {error && <p className="auth-error">{error}</p>}
          <p>Already Have an account? <Link to="/login">Click here</Link></p>
        </div>
      </div></>
  );
}
