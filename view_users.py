import sqlite3

conn = sqlite3.connect("artifacts/users.db")

cursor = conn.cursor()

cursor.execute("""
SELECT id, name, email
FROM users
""")

users = cursor.fetchall()

print("REGISTERED USERS")

for user in users:
    print(user)

conn.close()