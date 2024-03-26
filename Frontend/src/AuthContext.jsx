import React, { useState } from 'react';

const AuthContext = React.createContext({
  token: '',
  isLoggedIn: false,
  login: (token) => {},
  logout: () => {},
  images:[],
  addImage:(newImage)=>{},
  email:''
});

export const AuthContextProvider = (props) => {
  const initialState = localStorage.getItem('token');
  const [token, setToken] = useState(initialState);
  const [images,setImages] = useState([]);
  const [mail,setMail] = useState(localStorage.getItem('email'));

  const userIsLoggedIn = !!token;

  const loginHandler = (token, inputMail) => {
    setToken(token);
    setMail(inputMail);
    localStorage.setItem('token', token);
    localStorage.setItem('email', inputMail);
  };
  
  const logoutHandler = () => {
    setToken('');
    setMail('');
    localStorage.removeItem('token');
    localStorage.removeItem('email');
  };
  
  const imageHandler = (newImageList) =>{
    const sortedImageList = newImageList.slice().sort((a, b) => {
      // Sort images based on upload time in descending order
      return new Date(b.metadata.timeCreated) - new Date(a.metadata.timeCreated);
    }).map(image => image.url);
  
    setImages(sortedImageList);
    // setImages(newImageList);
  }


  const contextValue = {
    token: token,
    isLoggedIn: userIsLoggedIn,
    login: loginHandler,
    logout: logoutHandler,
    images:images,
    addImage:imageHandler,
    email:mail
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {props.children}
    </AuthContext.Provider>
  );
};

export default AuthContext;