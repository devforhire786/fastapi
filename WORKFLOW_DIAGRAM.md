# 🔄 Smart Lighting Application - Visual Workflow Diagrams

## 📊 **Main System Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   API Gateway   │───▶│  FastAPI App   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  WebSocket      │◀───│  Real-time      │◀───│  Business      │
│  Broadcasting   │    │  Updates        │    │  Logic         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Client Apps    │    │  MongoDB        │    │  ML Model      │
│  (Mobile/Web)   │    │  Database       │    │  Handler       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🏠 **Room Setup Workflow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Create    │───▶│  Configure  │───▶│ Initialize  │───▶│   Ready     │
│   Room      │    │Preferences  │    │   Lights    │    │   State     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ POST        │    │ PATCH       │    │ Default     │    │ System      │
│ /rooms/     │    │ /rooms/{id} │    │ Light: OFF  │    │ Monitoring  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 🤖 **AI Motion Detection Workflow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Motion    │───▶│   Data      │───▶│   ML        │───▶│   Light     │
│  Detected   │    │Collection   │    │ Prediction  │    │  Control    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ POST        │    │ Time, Day,  │    │ Confidence  │    │ ON/OFF +    │
│ /motion/    │    │ Motion      │    │ Score       │    │ Brightness  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Store     │───▶│  Broadcast  │
                       │   Event     │    │  Update     │
                       └─────────────┘    └─────────────┘
```

---

## 💡 **Light Control Workflow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│  Validate   │───▶│  Update     │───▶│  Broadcast  │
│  Request    │    │  Input      │    │  Database   │    │  Change     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ PUT/POST    │    │ Pydantic    │    │ MongoDB     │    │ WebSocket   │
│ /lights/    │    │ Validation  │    │ Update      │    │ Message     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Return    │    │  Real-time  │
                       │  Response   │    │  UI Update  │
                       └─────────────┘    └─────────────┘
```

---

## ⏰ **Scheduling Workflow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Create    │───▶│   Store     │───▶│   Monitor   │───▶│   Execute   │
│  Schedule   │    │  Schedule   │    │   Time      │    │   Action    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ POST        │    │ MongoDB     │    │ Background  │    │ Light       │
│ /schedule/  │    │ Insert      │    │  Process    │    │ Control     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Log       │    │  Broadcast  │
                       │   Event     │    │  Schedule   │
                       └─────────────┘    └─────────────┘
```

---

## 📊 **Analytics Workflow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Request   │───▶│   Query     │───▶│   Process   │───▶│   Return    │
│  Analytics  │    │  Database   │    │   Data      │    │  Results    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ GET         │    │ MongoDB     │    │ Calculate   │    │ JSON        │
│ /analytics/ │    │ Aggregation │    │ Statistics  │    │ Response    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Cache     │    │  Client     │
                       │   Results   │    │  Display    │
                       └─────────────┘    └─────────────┘
```

---

## 🔄 **Real-time Communication Flow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Event     │───▶│  WebSocket  │───▶│  Broadcast  │───▶│  Client     │
│  Occurs     │    │   Server    │    │  Message    │    │  Update     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Motion,     │    │ Connection  │    │ All Active  │    │ Mobile App, │
│ Light, AI   │    │ Manager     │    │ Connections │    │ Web UI     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Log       │    │  Real-time  │
                       │   Event     │    │  Sync       │
                       └─────────────┘    └─────────────┘
```

---

## 🧠 **AI Learning Workflow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Collect   │───▶│   Analyze   │───▶│   Train     │───▶│   Improve   │
│    Data     │    │  Patterns   │    │   Model     │    │  Accuracy   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Motion,     │    │ Time-based  │    │ Update      │    │ Better      │
│ Light, AI   │    │ Behavior    │    │ Parameters  │    │ Predictions │
│ Events      │    │ Analysis    │    │ & Weights   │    │ & Control   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Store     │    │  Monitor    │
                       │   Metrics   │    │  Progress   │
                       └─────────────┘    └─────────────┘
```

---

## 🔧 **System Health Monitoring**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Monitor   │───▶│   Check     │───▶│   Assess    │───▶│   Report    │
│   System    │    │  Components │    │   Health    │    │   Status    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Background  │    │ Database,   │    │ Health      │    │ Dashboard,  │
│  Process    │    │ Model, API  │    │ Score       │    │ Alerts      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Log       │    │  User       │
                       │   Issues    │    │  Notification│
                       └─────────────┘    └─────────────┘
```

---

## 📱 **User Experience Flow**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   App       │───▶│   View      │───▶│   Control   │───▶│   Monitor   │
│   Launch    │    │   Status    │    │   Lights    │    │   Results   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Check       │    │ Room        │    │ Manual      │    │ Analytics   │
│ Connection  │    │ Overview    │    │ Control     │    │ Dashboard   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                │                   │
                                ▼                   ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   Real-time │    │  Historical │
                       │   Updates   │    │  Data       │
                       └─────────────┘    └─────────────┘
```

---

## 🎯 **Key Workflow Benefits**

1. **Automated Decision Making** → AI handles routine lighting decisions
2. **Real-time Responsiveness** → Immediate updates across all interfaces
3. **Data-Driven Insights** → Analytics provide optimization opportunities
4. **Scalable Architecture** → Easy to add new rooms and features
5. **User-Friendly Interface** → Simple setup and control
6. **Energy Efficiency** → Smart scheduling and AI optimization
7. **Reliable Operation** → Robust error handling and monitoring
8. **Future-Proof Design** → Extensible for new capabilities

---

These workflow diagrams show how your Smart Lighting Application creates a seamless, intelligent system that automatically manages lighting while providing comprehensive control and monitoring capabilities. 