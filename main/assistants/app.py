from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    unique_id = db.Column(db.String(50), unique=True, nullable=False)

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    data = db.Column(db.Text, nullable=False)

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    new_user = User(name=data['name'], unique_id=data['unique_id'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/users/<int:id>', methods=['GET'])
def get_user(id):
    user = User.query.get_or_404(id)
    return jsonify({'name': user.name, 'unique_id': user.unique_id})

@app.route('/schedules', methods=['POST'])
def create_schedule():
    data = request.get_json()
    new_schedule = Schedule(
        user_id=data['user_id'],
        title=data['title'],
        start_time=datetime.strptime(data['start_time'], '%Y-%m-%d %H:%M:%S'),
        end_time=datetime.strptime(data['end_time'], '%Y-%m-%d %H:%M:%S')
    )
    db.session.add(new_schedule)
    db.session.commit()
    return jsonify({'message': 'Schedule created successfully'}), 201

@app.route('/schedules/<int:user_id>', methods=['GET'])
def get_schedules(user_id):
    schedules = Schedule.query.filter_by(user_id=user_id).all()
    return jsonify([{
        'title': s.title,
        'start_time': s.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': s.end_time.strftime('%Y-%m-%d %H:%M:%S')
    } for s in schedules])

@app.route('/timetables', methods=['POST'])
def create_timetable():
    data = request.get_json()
    new_timetable = Timetable(user_id=data['user_id'], data=data['data'])
    db.session.add(new_timetable)
    db.session.commit()
    return jsonify({'message': 'Timetable created successfully'}), 201

@app.route('/timetables/<int:user_id>', methods=['GET'])
def get_timetable(user_id):
    timetable = Timetable.query.filter_by(user_id=user_id).first_or_404()
    return jsonify({'data': timetable.data})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)