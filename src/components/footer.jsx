import React from 'react';
import { render } from 'react-dom';
import { FaCreativeCommons } from 'react-icons/fa';

export default class Footer extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-text-center uk-text-small">
        <span>
          &copy; Copyright 2024{' '}
          <a
            href="https://www.kc-ml2.com/en"
            target="_blank"
          >
            KC Machine Learning Lab
          </a>{' '}
        </span>

        <p>
          Powered by <FaCreativeCommons />{' '}
          <a
            href="https://github.com/denkiwakame/academic-project-template"
            target="_blank"
          >
            {' '}
            Academic Project Page Template{' '}
          </a>
        </p>
      </div>
    );
  }
}
