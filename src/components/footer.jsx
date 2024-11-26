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
        {/* <span>
          This template is provided under the
          <a
            href="https://creativecommons.org/licenses/by-sa/4.0/"
            target="_blank"
          >
            {' '}
            Attribution-ShareAlike 4.0 International (CC BY-SA 4.0){' '}
          </a>
          license.
        </span>
        <br /> */}
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
