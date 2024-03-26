import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { auth, userDB } from "../../config"; // Import db from your Firebase config
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { addDoc, collection } from "firebase/firestore"; // Import Firestore functions
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faEye, faEyeSlash } from "@fortawesome/free-solid-svg-icons";
import './Auth.css';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState(null);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleSignup = () => {
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        const user = userCredential.user;
        updateProfile(user, { displayName: username })
          .then(() => {
            console.log("User profile updated");
          })
          .catch((error) => {
            console.error("Error updating profile:", error);
          });

        // Add username and user ID to Firestore collection
        addDoc(collection(userDB, "users"), {
          username: username,
          userId: user.uid,
          email:user.email
        })
        .then(() => {
          console.log("Username and User ID added to Firestore");
        })
        .catch((error) => {
          console.error("Error adding document: ", error);
        });

        console.log("User Name:", username);
        console.log("User signed up:", user.email);
        navigate('/login');
      })
      .catch((error) => {
        setError(error.message);
      });
  };

  const togglePasswordVisibility = () => {
    setShowPassword((prevState) => !prevState);
  };



  return (
    <>
      <div className="nav">
      <img src="https://res.cloudinary.com/dcfzq326i/image/upload/v1711189350/tvbbxznhzh1byutclut1.png" alt="Logo"></img>
        <span className="navTitle">PicProbe</span>
        <button className="navButton" onClick={(e)=>navigate('/login')}>Login</button>
      </div>
      <div class="page-content">
            <h1 class="page-title">Visual Question and Answering System</h1>
            <p class="page-description">The web page can help users to respond to the queries related to the image by effectively utilizing 
             the textual information available in the images to offer additional useful cues and improve understanding of the visual elements in the content. 
             This system can precisely provide answers by properly interpreting the image features.</p>
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
         <div  className="password-container">
          <input
            type={showPassword ? "text" : "password"} // Toggle password visibility
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button
            className="toggle-password-button"
            onClick={togglePasswordVisibility}
          >
                <FontAwesomeIcon icon={showPassword ? faEye : faEyeSlash} />
          </button>
          </div>
          <button className="auth-button" onClick={handleSignup}>Sign Up</button>
          {error && <p className="auth-error">{error}</p>}
          <p>Already have an account? <Link to="/login">Click here</Link></p>
        </div>
      </div>
    </>
  );
}
