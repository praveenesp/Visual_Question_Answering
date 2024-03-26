import React, { useState, useEffect, useContext } from 'react';
import { useNavigate, Navigate } from 'react-router-dom';
import { collection, getDocs } from 'firebase/firestore';
import AuthContext from "../../AuthContext";
import { userDB } from '../../config'; // Import your Firebase auth and Firestore instance
import "./Nav.css";

export default function Nav() {
    const [logout, setLogout] = useState(false);
    const [userData, setUserData] = useState(null);
    const [showUserInfo, setShowUserInfo] = useState(false); // State to toggle user info display
    const navigate = useNavigate();
    const authCtx = useContext(AuthContext);

    useEffect(() => {
      const fetchData = async () => {
        try {
          const dataRef = collection(userDB, 'users');
          const querySnapshot = await getDocs(dataRef);
  
          const retrievedData = [];
          querySnapshot.forEach((doc) => {
            const { username, email ,userId } = doc.data();
            retrievedData.push({ username, email, userId, id: doc.id });
          });
  
          const userData = retrievedData.find(item => item.email === authCtx.email);
  
          setUserData(userData);
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };
  
      fetchData();
    }, [authCtx.email]);

    function handleLogout() { 
        localStorage.clear();
        setLogout(true);
    }

    if (logout) {
        return <Navigate to="/login" />;
    }

    console.log(userData);
    return (
        <div className="nav">
            <img src="https://res.cloudinary.com/dcfzq326i/image/upload/v1711189350/tvbbxznhzh1byutclut1.png" alt="Logo" />
            <span className="navTitle">PicProbe</span>
           
            <button className="navButton" onClick={() => navigate("/home")}>Home</button>
            <button className="navButton" onClick={() => navigate("/history")}>History</button>
            <button className="navButton" onClick={handleLogout}>Logout</button>
            <button className="navButton" onClick={() => setShowUserInfo(prevState => !prevState)}>
                {showUserInfo ? 'Hide Info' : 'Show Info'}
            </button>
            {showUserInfo && userData && (
                <div className="user-info-card">
                    <h3>User Information</h3>
                    <p><strong>Username:</strong> {userData.username}</p>
                    <p><strong>Email:</strong> {userData.email}</p>
                    {/* <p><strong>UserID:</strong> {userData.userId}</p> */}
                </div>
            )}
        </div>
    );
}
