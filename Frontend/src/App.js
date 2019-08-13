import React, { Component } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.css';
import Particles from 'react-particles-js';
import swal from 'sweetalert';
import {Link} from 'react-router-dom';

const particlesOptions = {
  particles: {
    number: {
      value: 60,
      density: {
        enable: true,
        value_area: 800
      }
    }
  }
}
class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      isLoading: false,
      done: false,
      formData: {
        pregVal: '',
        plasVal: '',
        skinVal: '',
        insuVal: '',
        massVal: '',
        pediVal: '',
        Age: ''
      },
      result: ""
    };
  }

  handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    var formData = this.state.formData;
    formData[name] = value;
    this.setState({
      formData
    });
  }

  handlePredictClick = (event) => {
    const formData = this.state.formData;
    this.setState({ isLoading: true });
    fetch('http://127.0.0.1:5000/prediction/', 
      {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify(formData)
      })
      .then(response => response.json())
      .then(response => {
        this.setState({
          result: response.result,
          isLoading: false
        });
      });
  }

  handleCancelClick = (event) => {
    this.setState({ result: "" });
    this.setState({done: true})
  }

  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
   // const result = this.state.result;

    
    let x = null;
    let y = null;

    if(this.state.isLoading === true)
 { 
    x =swal({
      text: this.state.result,
      icon: "success",

    })
  }
  

    return (
    
      <Container>
         <Particles className='particles'
          params={particlesOptions}
        />
         <article className="br3 ba b--black-20 mv6 w-100 w-120-m w-25-l mw6 shadow-3 center">
        <main className="pa4 black-200">
          <div className="measure">
            <fieldset id="sign_up" className="ba b--transparent ph0 mh0">
              <legend className="f1 fw6 ph0 mh0">&nbsp;&nbsp;Diabetes Predictor</legend>
              <div className="mv3">
              <Form>
            <Form.Row>
              <Form.Group as={Col} sm="3">
                <Form.Label><p class="b">pregVal</p></Form.Label>
                <Form.Control 
                  type="text"
                  value={formData.pregVal}
                  name="pregVal"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3">
                <Form.Label><p class="b">plasVal</p></Form.Label>
                <Form.Control 
                type="text"
                  value={formData.plasVal}
                  name="plasVal"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3">
                <Form.Label><p class="b">massVal</p></Form.Label>
                <Form.Control 
                  type="text"
                  value={formData.massVal}
                  name="massVal"
                  onChange={this.handleChange}>
                
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3">
                <Form.Label><p class="b">insuVal</p></Form.Label>
                <Form.Control 
                 type="text"
                  value={formData.insuVal}
                  name="insuVal"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3">
                <Form.Label><p class="b">Age</p></Form.Label>
                <Form.Control 
                 type="text"
                  value={formData.Age}
                  name="Age"
                  onChange={this.handleChange}>
        
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3"> 
                <Form.Label><p class="b">skinVal</p></Form.Label>
                <Form.Control 
                  type="text"
                  value={formData.skinVal}
                  name="skinVal"
                  onChange={this.handleChange}>
              
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3">
                <Form.Label><p class="b">pediVal</p></Form.Label>
                <Form.Control 
                type="text"
                  value={formData.pediVal}
                  name="pediVal"
                  onChange={this.handleChange}>
              
                </Form.Control>
              </Form.Group>
              <Form.Group as={Col}  sm="3">
                <Form.Label><p class="b">presVal</p></Form.Label>
                <Form.Control 
                type="text"
                  value={formData.presVal}
                  name="presVal"
                  onChange={this.handleChange}>
                </Form.Control>
              </Form.Group>
            </Form.Row>
            <Row>
              <Col>
                <Button
                  block
                  variant="success"
                  disabled={isLoading}
                  onClick={!isLoading ? this.handlePredictClick : null}>
                  { isLoading ? 'Making prediction' : 'Predict' }
                </Button>
                <p class="b mw5 mw7-ns center">{this.state.result}</p>
              </Col>
              <Col>
                <Button
                  block
                  variant="danger"
                  disabled={isLoading}
                  onClick={this.handleCancelClick}>
                  Reset prediction
                </Button>
              </Col>
            </Row>
          </Form>
              </div>
            </fieldset>  
          </div>
        </main>
      </article>
      </Container>
    );
  }
}

export default App;