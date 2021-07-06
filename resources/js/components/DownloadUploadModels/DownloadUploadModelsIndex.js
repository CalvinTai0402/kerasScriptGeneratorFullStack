import React from 'react';
import ServerTable from 'react-strap-table';
import { AiFillDelete, AiFillEdit, AiFillPlusSquare, AiFillMinusSquare, AiOutlineDownload } from "react-icons/ai";
import { Link } from "react-router-dom";
import Spinner from "../Spinner";
import axios from 'axios';
import _ from 'lodash';

class DownloadUploadModelsIndex extends React.Component {
    state = {
        selectedKerasModels: [],
        kerasModelsIDs: [],
        isAllChecked: false,
        deleting: false,
    };

    check_all = React.createRef();

    handleCheckboxTableChange = (event) => {
        const value = event.target.value;
        let selectedKerasModels = this.state.selectedKerasModels.slice();

        selectedKerasModels.includes(value) ?
            selectedKerasModels.splice(selectedKerasModels.indexOf(value), 1) :
            selectedKerasModels.push(value);

        this.setState({ selectedKerasModels: selectedKerasModels }, () => {
            this.check_all.current.checked = _.difference(this.state.kerasModelsIDs, this.state.selectedKerasModels).length === 0;
        });
    }

    handleCheckboxTableAllChange = (event) => {
        this.setState({ selectedKerasModels: [...new Set(this.state.selectedKerasModels.concat(this.state.kerasModelsIDs))] }, () => {
            this.check_all.current.checked = _.difference(this.state.kerasModelsIDs, this.state.selectedKerasModels).length === 0;
        });
    }

    handleDelete = async (id) => {
        let password = prompt("Please enter the password to delete");
        if (password !== process.env.MIX_DELETE_KEY) {
            alert("You are not allowed to delete this!")
        } else {
            this.setState({ deleting: true })
            const res = await axios.delete(`kerasModel/${id}`);
            if (res.data.status === 204) {
                this.setState({ deleting: false })
            }
        }
    };

    handleDeleteMany = async () => {
        let password = prompt("Please enter the password to delete");
        if (password !== process.env.MIX_DELETE_KEY) {
            alert("You are not allowed to delete this!")
        } else {
            this.setState({ deleting: true })
            const { selectedKerasModels } = this.state
            let selectedKerasModelIds = selectedKerasModels.map(Number);
            const res = await axios.post(`kerasModel/deleteMany`, {
                selectedKerasModelIds: selectedKerasModelIds
            });
            if (res.data.status === 204) {
                this.setState({ deleting: false })
            }
        }
    }

    handleDownload = async (downloadId) => {
        const res = await axios.get(`kerasModel/${downloadId}/downloadFile`, {
            responseType: 'blob',
        });
        const url = window.URL.createObjectURL(new Blob([res.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'kerasModel.zip');
        document.body.appendChild(link);
        link.click();
    }

    render() {
        const { deleting } = this.state;
        let self = this;
        let url = ""
        console.log(process.env.MIX_API_URL)
        if (process.env.MIX_APP_ENV === "production") {
            url = `${process.env.MIX_API_URL}/kerasModel`;
        } else {
            url = `http://127.0.0.1:8000/kerasModel`;
        }
        const columns = ['id', 'file_name', 'description', 'kerasModelFile', 'actions']
        let checkAllInput = (<input type="checkbox" ref={this.check_all} onChange={this.handleCheckboxTableAllChange} />);
        const options = {
            perPage: 10,
            headings: { id: checkAllInput },
            sortable: ['file_name', 'description'],
            requestParametersNames: { query: 'search', direction: 'order' },
            responseAdapter: function (res) {
                let kerasModelsIDs = res.data.map(a => a.id.toString());
                self.setState({ kerasModelsIDs: kerasModelsIDs }, () => {
                    self.check_all.current.checked = _.difference(self.state.kerasModelsIDs, self.state.selectedKerasModels).length === 0;
                });

                return { data: res.data, total: res.total }
            },
            texts: {
                show: 'Keras Models'
            },
        };
        return (
            <div>
                <button className="btn btn-primary create" style={{ marginRight: "8px" }}>
                    <Link to={'downloadUploadModels/create'}>
                        <div style={{ color: "white" }} >
                            <AiFillPlusSquare color="white" size="20" />
                            <span style={{ marginLeft: "8px" }} >
                                Create
                            </span>
                        </div>
                    </Link>
                </button>
                <button className="btn btn-danger delete" onClick={() => { self.handleDeleteMany() }}>
                    <div style={{ color: "white" }} >
                        <AiFillMinusSquare color="white" size="20" />
                        <span style={{ marginLeft: "8px" }} >
                            Delete Many
                        </span>
                    </div>
                </button>
                {
                    deleting ? <Spinner /> :
                        <ServerTable columns={columns} url={url} options={options} bordered hover updateUrl>
                            {
                                function (row, column) {
                                    switch (column) {
                                        case 'id':
                                            return (
                                                <input key={row.id.toString()} type="checkbox" value={row.id.toString()}
                                                    onChange={self.handleCheckboxTableChange}
                                                    checked={self.state.selectedKerasModels.includes(row.id.toString())} />
                                            );
                                        case 'actions':
                                            return (
                                                // <div style={{ display: "inline-block", justifyContent: "space-between" }}>
                                                <div style={{ textAlign: 'center' }}>
                                                    <button className="btn btn-primary" style={{ marginRight: "5px" }}>
                                                        <Link to={'downloadUploadModels/' + row.id + '/edit'}>
                                                            <AiFillEdit color="white" />
                                                            <div style={{ color: "white" }} >
                                                                Edit
                                                            </div>
                                                        </Link>
                                                    </button>
                                                    <button className="btn btn-danger" style={{ marginLeft: "5px" }} onClick={() => { self.handleDelete(row.id) }}>
                                                        <AiFillDelete color="white" />
                                                        <div style={{ color: "white" }}>
                                                            Delete
                                                        </div>
                                                    </button>
                                                    <button className="btn btn-success" style={{ marginLeft: "5px" }} onClick={() => { self.handleDownload(row.id) }}>
                                                        <AiOutlineDownload color="white" />
                                                        <div style={{ color: "white" }}>
                                                            Download
                                                        </div>
                                                    </button>
                                                </div>

                                            );
                                        default:
                                            return (row[column]);
                                    }
                                }
                            }
                        </ServerTable >
                }</div>
        );
    }
}

export default DownloadUploadModelsIndex;