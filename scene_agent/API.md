# ADK API — sessão + run

Base: `https://adk.akcit.fun`

`appName` = `env_agent`

---

## 1. Criar sessão

```bash
curl -X POST https://adk.akcit.fun/apps/env_agent/users/u_123/sessions/s_123 \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Resposta:**

```json
{
  "id": "s_123",
  "appName": "env_agent",
  "userId": "u_123",
  "state": {},
  "events": [],
  "lastUpdateTime": 1743711430.022186
}
```

---

## 2. Enviar mensagem

```bash
curl -X POST https://adk.akcit.fun/run \
  -H "Content-Type: application/json" \
  -d '{
    "appName": "env_agent",
    "userId": "u_123",
    "sessionId": "s_123",
    "newMessage": {
      "role": "user",
      "parts": [{"text": "O que tem em cima da minha mesa?"}]
    }
  }'
```

**Resposta** (array de eventos; pode incluir chamadas de tool no meio):

```json
[
  {
    "content": {
      "parts": [{"functionCall": {"name": "query_environment_objects", "args": {"query": "objetos em cima da mesa"}}}],
      "role": "model"
    },
    "author": "env_agent"
  },
  {
    "content": {
      "parts": [{"functionResponse": {"name": "query_environment_objects", "response": {"status": "ok", "hits": []}}}],
      "role": "user"
    },
    "author": "env_agent"
  },
  {
    "content": {
      "parts": [{"text": "Na mesa há um notebook, uma caneca e um monitor."}],
      "role": "model"
    },
    "author": "env_agent"
  }
]
```

**Resposta do LLM:** último objeto do array com `"role": "model"` → campo `content.parts[].text`.

No exemplo acima: `"Na mesa há um notebook, uma caneca e um monitor."`

---

## Unity (C#)

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

// --- sessão (resposta) ---
[Serializable]
public class SessionResponse
{
    public string id;
    public string appName;
    public string userId;
    public Dictionary<string, object> state;
    public List<object> events;
    public double lastUpdateTime;
}

// --- run (request) ---
[Serializable]
public class RunRequest
{
    public string appName = "env_agent";
    public string userId;
    public string sessionId;
    public Message newMessage;
}

[Serializable]
public class Message
{
    public string role = "user";
    public List<Part> parts;
}

[Serializable]
public class Part
{
    public string text;
}

// --- run (resposta) ---
[Serializable]
public class RunEvent
{
    public Content content;
    public string author;
}

[Serializable]
public class Content
{
    public string role; // "model" = resposta do LLM
    public List<Part> parts;
}

// extrair texto do LLM
public static string GetLlmText(RunEvent[] events) =>
    events?.LastOrDefault(e => e.content?.role == "model")
        ?.content?.parts?.FirstOrDefault(p => !string.IsNullOrEmpty(p.text))?.text;

// exemplo de uso (coroutine)
IEnumerator CreateSessionAndRun(string userId, string sessionId, string prompt)
{
    const string baseUrl = "https://adk.akcit.fun";

    // 1. criar sessão
    var sessionUrl = $"{baseUrl}/apps/env_agent/users/{userId}/sessions/{sessionId}";
    using (var req = new UnityWebRequest(sessionUrl, "POST"))
    {
        req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{}"));
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        yield return req.SendWebRequest();
        // SessionResponse session = JsonConvert.DeserializeObject<SessionResponse>(req.downloadHandler.text);
    }

    // 2. run
    var runBody = JsonConvert.SerializeObject(new RunRequest
    {
        userId = userId,
        sessionId = sessionId,
        newMessage = new Message { parts = new List<Part> { new Part { text = prompt } } }
    });
    using (var req = new UnityWebRequest($"{baseUrl}/run", "POST"))
    {
        req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(runBody));
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        yield return req.SendWebRequest();

        var events = JsonConvert.DeserializeObject<RunEvent[]>(req.downloadHandler.text);
        string llmReply = GetLlmText(events); // <-- isso é o que importa
    }
}
```
