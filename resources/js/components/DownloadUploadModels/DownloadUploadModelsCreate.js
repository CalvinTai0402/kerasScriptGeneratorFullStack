import React, { Component } from 'react';
import {
    Grid,
    Form,
    Segment,
    Button,
    Header,
    Message,
} from "semantic-ui-react";
import axios from 'axios';

class DownloadUploadModelsCreate extends Component {
    state = {
        fileName: "",
        description: "",
        errors: [],
        loading: false,
        kerasModelFile: "",
    }

    handleChange = event => { this.setState({ [event.target.name]: event.target.value }); };

    handleStore = async event => {
        event.preventDefault();
        const { fileName, description, kerasModelFile } = this.state;
        const data = new FormData()
        data.append('file_name', fileName)
        data.append('description', description)
        data.append('kerasModelFile', kerasModelFile)
        if (this.isFormValid(this.state)) {
            this.setState({ loading: true });
            const res = await axios.post('/kerasModel', data).catch((e) => {
                console.log(e);
            });
            console.log(res)
            if (res.data.status === 201) {
                this.setState({ loading: false });
                this.props.history.push("/downloadUploadModels");
            }
        }
    };

    handleFileChange = (e) => {
        this.setState({
            kerasModelFile: e.target.files[0]
        })
    }

    displayErrors = errors => errors.map((error, i) => <p key={i}>{error}</p>);

    handleInputError = (errors, inputName) => {
        return errors.some(error => error.toLowerCase().includes(inputName)) ? "error" : "";
    };

    isFormValid = ({ fileName, description, kerasModelFile }) => {
        if (fileName && description && kerasModelFile && kerasModelFile.type === "application/x-zip-compressed") { return true }
        this.setState({ errors: [] }, () => {
            const { errors } = this.state;
            if (fileName.length === 0) {
                errors.push("File name cannot be empty")
            }
            if (description.length === 0) {
                errors.push("Description cannot be empty")
            }
            if (kerasModelFile.length === 0) {
                errors.push("Keras model file cannot be empty")
            } else if (kerasModelFile.type !== "application/x-zip-compressed") {
                errors.push("Keras model file must be type: zip")
            }
            this.setState({ errors })
        });
    };

    render() {
        const { fileName, description, errors, loading } = this.state;
        return (
            <div>
                <Grid textAlign="center" verticalAlign="middle" className="app">
                    <Grid.Column style={{ maxWidth: 450 }}>
                        <Header as="h1" icon color="blue" textAlign="center">
                            Create Keras Model
                        </Header>
                        <Form id="myForm" onSubmit={this.handleStore} size="large">
                            <Segment stacked>
                                <Form.Field>
                                    <label>File Name</label>
                                    <Form.Input
                                        fluid
                                        name="fileName"
                                        onChange={this.handleChange}
                                        value={fileName}
                                        className={this.handleInputError(errors, "name")}
                                    />
                                </Form.Field>
                                <Form.Field>
                                    <label>Description</label>
                                    <Form.Input
                                        fluid
                                        name="description"
                                        onChange={this.handleChange}
                                        value={description}
                                        className={this.handleInputError(errors, "description")}
                                    />
                                </Form.Field>
                                <Form.Field>
                                    <Form.Input
                                        id="file"
                                        type="file"
                                        className={this.handleInputError(errors, "keras")}
                                        onChange={this.handleFileChange} />
                                </Form.Field>
                                <Button
                                    disabled={loading}
                                    className={loading ? "loading" : ""}
                                    color="blue"
                                    fluid
                                    size="large"
                                >
                                    Create Keras Model
                                </Button>
                            </Segment>
                        </Form>
                        {errors.length > 0 && (
                            <Message error>
                                <h3>Error</h3>
                                {this.displayErrors(errors)}
                            </Message>
                        )}
                    </Grid.Column>
                </Grid>
            </div>
        );
    }
}

export default DownloadUploadModelsCreate;