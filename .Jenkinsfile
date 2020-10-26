pipeline {
    agent {
        docker {
            image 'ubuntu_tester'
            args '-u root:root -v ${HOME}/html/docs:/docs -v ${HOME}/html/_ci:/ci'
        }
    }
    environment {
        PJ_NAME = 'fastHan'
        POST_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/3aa3a3a1-88f2-4c36-853b-21361a8f1234'
    }
    stages {
        stage('Package Installation') {
            steps {
                sh 'python setup.py install'
            }
        }
        stage('Parallel Stages') {
            parallel {
                stage('Document Building') {
                    steps {
                        sh 'cd docs && make prod'
                        sh 'rm -rf /docs/${PJ_NAME}'
                        sh 'mv docs/build/html /docs/${PJ_NAME}'
                    }
                }
                stage('Package Testing') {
                    steps {
                        sh 'pytest ./test --html=test_results.html --self-contained-html'
                    }
                }
            }
        }
    }
    post {
        always {
            sh 'post'
        }

    }

}
