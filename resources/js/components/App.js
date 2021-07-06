import React from 'react';
import ReactDOM from 'react-dom';
import Sidebar from './Sidebar'
import 'semantic-ui-css/semantic.min.css'

function App() {
  return (
    <div>
      <Sidebar />
    </div>
  );
}

export default App;

if (document.getElementById('root')) {
  ReactDOM.render(<App />, document.getElementById('root'));
}
