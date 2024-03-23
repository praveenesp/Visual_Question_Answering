import { useState } from "react";
import { Link, useNavigate } from "react-router-dom"; // Import useNavigate
import { auth } from "../../config";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import './Auth.css';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate(); // Initialize navigate hook

  const handleSignup = () => {
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        const user = userCredential.user;
        updateProfile(user, { displayName: username })
          .then(() => {
            console.log("User profile updated ");
          })
          .catch((error) => {
            console.error("Error updating profile:", error);
          });
        console.log("User Name:", username);  
        console.log("User signed up:", user.email);
        navigate('/login'); // Navigate to login page
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  return (
    <>
      <div className="nav">
      <img src="https://res.cloudinary.com/dcfzq326i/image/upload/v1711189350/tvbbxznhzh1byutclut1.png" alt="Logo"></img>
        <span className="navTitle">PicProbe</span>
        <button className="navButton" onClick={(e)=>navigate('/login')}>Login</button>
      </div>
      <div className="auth-container">
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
          <p>Already have an account? <Link to="/login">Click here</Link></p>
        </div>
      </div>
    </>
  );
}
