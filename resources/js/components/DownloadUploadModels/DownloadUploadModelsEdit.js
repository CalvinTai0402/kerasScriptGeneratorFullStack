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
        loading: false
    }

    handleChange = event => { this.setState({ [event.target.name]: event.target.value }); };

    handleUpdate = async event => {
        event.preventDefault();
        let password = prompt("Please enter the password to update");
        if (password !== process.env.MIX_UPDATE_KEY) {
            alert("You are not allowed to update this!")
        } else {
            const { fileName, description } = this.state;
            if (this.isFormValid(this.state)) {
                this.setState({ loading: true });
                const id = this.props.match.params.id;
                const res = await axios.put(`/kerasModel/${id}`, {
                    file_name: fileName,
                    description: description
                });
                if (res.data.status === 200) {
                    this.setState({ loading: false });
                    this.props.history.push("/downloadUploadModels");
                }
            }
        }
    };

    displayErrors = errors => errors.map((error, i) => <p key={i}>{error}</p>);

    handleInputError = (errors, inputName) => {
        return errors.some(error => error.toLowerCase().includes(inputName)) ? "error" : "";
    };

    isFormValid = ({ fileName, description }) => {
        if (fileName && description) { return true }
        this.setState({ errors: [] }, () => {
            const { errors } = this.state;
            if (fileName.length === 0) {
                errors.push("File name cannot be empty")
            }
            if (description.length === 0) {
                errors.push("Description cannot be empty")
            }
            this.setState({ errors })
        });
    };

    async componentDidMount() {
        const id = this.props.match.params.id;
        const res = await axios.get(`/kerasModel/${id}/edit`);
        this.setState({ fileName: res.data.kerasModel.file_name });
        this.setState({ description: res.data.kerasModel.description });
    }

    render() {
        const { fileName, description, errors, loading } = this.state;
        return (
            <div>
                <Grid textAlign="center" verticalAlign="middle" className="app">
                    <Grid.Column style={{ maxWidth: 450 }}>
                        <Header as="h1" icon color="blue" textAlign="center">
                            Edit Keras Model
                        </Header>
                        <Form onSubmit={this.handleUpdate} size="large">
                            <Segment stacked>
                                <Form.Field>
                                    <label>File Name</label>
                                    <Form.Input
                                        fluid
                                        name="fileName"
                                        onChange={this.handleChange}
                                        value={fileName}
                                        className={this.handleInputError(errors, "file")}
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
                                <Button
                                    disabled={loading}
                                    className={loading ? "loading" : ""}
                                    color="blue"
                                    fluid
                                    size="large"
                                >
                                    Update
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