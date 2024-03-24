import { auth, provider } from "../../config";
import { useState, useEffect } from "react";
import { Link ,useNavigate} from 'react-router-dom';
import { signInWithPopup, signInWithEmailAndPassword } from "firebase/auth";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faEye, faEyeSlash } from "@fortawesome/free-solid-svg-icons";
import './Auth.css';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleGoogleSignIn = () => {
    signInWithPopup(auth, provider)
      .then((result) => {
        setUser(result.user);
        localStorage.setItem("email", result.user.email);
      })
      .catch((error) => {
        setError(error.message);
      });
      navigate('/home');
  };

  const handleEmailSignIn = () => {
    signInWithEmailAndPassword(auth, email, password)
      .then((result) => {
        setUser(result.user);
        localStorage.setItem("email", result.user.email);
        navigate('/home');
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
  const togglePasswordVisibility = () => {
    setShowPassword((prevState) => !prevState);
  };



  return (
    <div>
      <div className="nav">
      <img src="https://res.cloudinary.com/dcfzq326i/image/upload/v1711189350/tvbbxznhzh1byutclut1.png" alt="Logo"></img>
      <span className="navTitle">PicProbe</span>
      <button className="navButton" onClick={(e)=>navigate("/")}>Singup</button>
      </div>
      <div class="page-content">
            <h1 class="page-title">Visual Question and Answering System</h1>
            <p class="page-description">The web page can help users to respond to the queries related to the image by effectively utilizing 
             the textual information available in the images to offer additional useful cues and improve understanding of the visual elements in the content. 
             This system can precisely provide answers by properly interpreting the image features.</p>
            </div>
          <div className="auth-container">
              <h2>Sign In</h2>
              <div className="auth-form">
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
                <button className="auth-button" onClick={handleEmailSignIn}>Sign In with Email</button>
                <p className="reset-password-text">Forgot your password? <Link to="/reset-password">Reset it here.</Link></p>
                <p className="or-text">------ OR ------</p>
                <button className="auth-button" onClick={handleGoogleSignIn}>Sign In with Google</button>
                <p className="sign-up-text">Don't have an account? <Link to="/">Sign up</Link></p>
                {error && <p className="auth-error">{error}</p>}
              </div>
            </div>
    </div>
  );
}
