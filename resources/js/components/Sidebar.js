import React, { Fragment } from "react";
import './App.css';
import ScriptGenerator from './Generators/ScriptGenerator'
import ExampleGenerator from './Generators/ExampleGenerator'
import UtilitiesGenerator from './Generators/UtilitiesGenerator'
import Footer from './Footer';
import { ProSidebar, Menu, MenuItem, SidebarHeader, SidebarContent } from 'react-pro-sidebar';
import 'react-pro-sidebar/dist/css/styles.css';
import { FaBattleNet, FaFreebsd, FaBity, FaJoget } from "react-icons/fa";
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link
} from "react-router-dom";
import DownloadUploadModelsIndex from "./DownloadUploadModels/DownloadUploadModelsIndex"
import DownloadUploadModelsCreate from "./DownloadUploadModels/DownloadUploadModelsCreate"
import DownloadUploadModelsEdit from "./DownloadUploadModels/DownloadUploadModelsEdit"

class Sidebar extends React.Component {
    state = {
        menuCollapse: true
    };

    menuIconClick = () => {
        const { menuCollapse } = this.state;
        menuCollapse ? this.setState({ menuCollapse: false }) : this.setState({ menuCollapse: true })
    };

    render() {
        const { menuCollapse } = this.state;
        return (
            <div >
                <Router basename="/">
                    <div id="sidebar" style={{ display: 'grid', gridTemplateColumns: '200px auto' }}>
                        <ProSidebar className='sideBar' collapsed={menuCollapse}>
                            <SidebarHeader className="sideBarHeader">
                                <p className="clickable" onClick={this.menuIconClick}>{menuCollapse ? "K.S.G." : "Keras Script Generator"}</p>
                            </SidebarHeader>
                            <SidebarContent>
                                <Menu iconShape="square">
                                    <MenuItem icon={<FaBattleNet />}>
                                        Model Building
                                        <Link to="/model" />
                                    </MenuItem>
                                    <MenuItem icon={<FaFreebsd />}>
                                        Examples
                                        <Link to="/examples" />
                                    </MenuItem>
                                    <MenuItem icon={<FaBity />}>
                                        Utilities
                                        <Link to="/utilities" />
                                    </MenuItem>
                                    <MenuItem icon={<FaJoget />}>
                                        Available models
                                        <Link to="/downloadUploadModels?search=&limit=10&page=1&orderBy=&order=desc" />
                                    </MenuItem>
                                </Menu>
                            </SidebarContent>

                        </ProSidebar>
                        <div className="centerVandH">
                            <Switch>
                                <Route path="/model">
                                    <ScriptGenerator />
                                    <Footer />
                                </Route>
                                <Route path="/examples">
                                    <ExampleGenerator />
                                    <Footer />
                                </Route>
                                <Route path="/utilities">
                                    <UtilitiesGenerator />
                                    <Footer />
                                </Route>
                                <Route className="centerVandH" exact path="/downloadUploadModels" render={(props) => <Fragment><DownloadUploadModelsIndex {...props} /><Footer /></Fragment>} />
                                <Route exact path="/downloadUploadModels/create" render={(props) => <Fragment><DownloadUploadModelsCreate {...props} /><Footer /></Fragment>} />
                                <Route exact path="/downloadUploadModels/:id/edit" render={(props) => <Fragment><DownloadUploadModelsEdit {...props} /><Footer /></Fragment>} />
                                <Route path="/">
                                    <ScriptGenerator />
                                    <Footer />
                                </Route>
                            </Switch>
                        </div>
                    </div>
                </Router>
            </div>
        );
    }
}

export default Sidebar;
