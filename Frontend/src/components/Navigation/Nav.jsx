
import { useState } from 'react';
import "./Nav.css"
import { Link,useNavigate,Navigate} from 'react-router-dom';

export default function Nav()
{
    const [logout, setLogout] = useState(false);
    const navigate = useNavigate();
    function handleLogout() { 
        localStorage.clear();
        setLogout(true);
    }

    if (logout) {
        return <Navigate to="/login" />;
    }
   return (
    <div className="nav">
      <img src="https://res.cloudinary.com/dcfzq326i/image/upload/v1711189350/tvbbxznhzh1byutclut1.png" alt="Logo"></img>
      <span className="navTitle">PicProbe</span>
      {/* <Link href="/" className="navLink">Home</Link>
      <Link href="/about" className="navLink">About Project</Link>
   <   Link href="/contact" className="navLink">Contact Us</Link>*/}
      {/* <Link href="/history" className="navLink">History</Link>  */}
      <button className="navButton" onClick={(e)=>navigate("/home")}>Home</button>
      <button className="navButton" onClick={(e)=>navigate("/history")}>History</button>
      <button className="navButton" onClick={handleLogout}>Logout</button>
  </div>
  
  );
};

