
import { useState } from 'react';
import "./Nav.css"
import { Link,Navigate} from 'react-router-dom';

export default function Nav()
{
    const [logout, setLogout] = useState(false);
    
    function handleLogout() { 
        localStorage.clear();
        setLogout(true);
    }

    if (logout) {
        return <Navigate to="/" />;
    }
   return (
    <div className="nav">
      <span className="navTitle">👁️ VQA</span>
      <Link href="/" className="navLink">Home</Link>
      <Link href="/about" className="navLink">About Project</Link>
      <Link href="/contact" className="navLink">Contact Us</Link>
      <Link href="/history" className="navLink">History</Link>
      <button className="navButton" onClick={handleLogout}>Logout</button>
    </div>
  );
};

