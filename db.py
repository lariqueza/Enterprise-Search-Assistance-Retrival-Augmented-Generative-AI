import pymongo
from pymongo.errors import ConnectionFailure


try:
    #create mongodb instnace and disable certificate verification
    client = pymongo.MongoClient(
        "",
        tls=True,
        tlsAllowInvalidCertificates=True
    )

    #verify connection
    client.admin.command('ping')
    print("Handshake successful connected to MongoDB Atlas")

    #access database
    db = client.droid

    #access collection
    collection = db.chatbot

except ConnectionFailure as e:
    print(f"could not connect to MongoDB: {e}")

except Exception as e:
    print(f"An error occured: {e}")








