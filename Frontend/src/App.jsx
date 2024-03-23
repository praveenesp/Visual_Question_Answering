import React, { useState } from 'react';
import Home from './components/Home/Home';
import Login from './components/Singin/Login';
import Signup from './components/Singin/Singup';
import ResetPassword from './components/Singin/ResetPassword';
import History from './components/History/History';
import { Routes, Route } from 'react-router-dom';
function App()
{
  return (
    <div>
       {/* <Signin/> */}
            <Routes>
               <Route path="/" element={<Signup />} />
                <Route path="/login" element={<Login />} />
                <Route path="/home" element={<Home/>}/>
                <Route path="/reset-password" element={<ResetPassword />} />
                <Route path="/history" element={<History/>}/>
            </Routes>
    </div>
  );
}

export default App;
