/**
 * Belaynish Telegram Bot - unified /ai command
 * - Chat (HuggingFace Space/API, Replicate fallback)
 * - Wiki, DuckDuckGo, Translate (@vitalets)
 * - Media: Runway + Replicate (unified under /ai media ...)
 * - TTS: ElevenLabs or Replicate Kokoro
 * - Redis memory (Upstash) fallback -> in-memory cache
 *
 * Notes:
 * - Answers are English by default.
 * - Every bot reply is prefixed with "Belaynish".
 */

import 'dotenv/config';
import { Telegraf } from 'telegraf';
import axios from 'axios';
import Redis from 'ioredis';
import NodeCache from 'node-cache';
import translateLib from '@vitalets/google-translate-api';

const BOT_TOKEN = process.env.TELEGRAM_TOKEN;
if (!BOT_TOKEN) {
  console.error('Missing TELEGRAM_TOKEN in env.');
  process.exit(1);
}
const bot = new Telegraf(BOT_TOKEN);

/* ---------------------------
   Memory (Redis or in-memory)
   --------------------------- */
const MEMORY_TTL = parseInt(process.env.MEMORY_TTL_SECONDS || '10800', 10); // seconds
let redis = null;
let usingRedis = false;
if (process.env.REDIS_URL || process.env.UPSTASH_REDIS_REST_URL) {
  try {
    const url = process.env.REDIS_URL || process.env.UPSTASH_REDIS_REST_URL;
    redis = new Redis(url, {
      password: process.env.UPSTASH_REDIS_REST_TOKEN || undefined
    });
    usingRedis = true;
    console.log('Using Redis memory.');
  } catch (e) {
    console.warn('Redis init failed, falling back to memory cache:', e.message);
    usingRedis = false;
  }
}
const memCache = new NodeCache({ stdTTL: MEMORY_TTL, checkperiod: 120 });

async function getMemory(chatId) {
  const key = `memory:${chatId}`;
  if (usingRedis && redis) {
    try {
      const v = await redis.get(key);
      return v ? JSON.parse(v) : [];
    } catch (e) {
      console.warn('Redis get failed:', e.message);
      usingRedis = false;
    }
  }
  return memCache.get(key) || [];
}
async function saveMemory(chatId, history) {
  const key = `memory:${chatId}`;
  if (usingRedis && redis) {
    try {
      await redis.set(key, JSON.stringify(history), 'EX', MEMORY_TTL);
      return;
    } catch (e) {
      console.warn('Redis set failed:', e.message);
      usingRedis = false;
    }
  }
  memCache.set(key, history);
}

/* ---------------------------
   Utilities
   --------------------------- */
const withPrefix = (text) => `Belaynish\n\n${text}`;

function safeFirst(arr, fallback = null) {
  return Array.isArray(arr) && arr.length ? arr[0] : fallback;
}

/* ---------------------------
   Hugging Face Space (free) and HF Inference API
   --------------------------- */
async function callHfSpace(prompt, spaceUrl = process.env.HF_SPACE_URL) {
  if (!spaceUrl) throw new Error('HF_SPACE_URL not configured');
  // Try /run/predict (gradio) and /api/predict (alt)
  const base = spaceUrl.replace(/\/$/, '');
  const tryUrls = [`${base}/run/predict`, `${base}/api/predict`, base];
  for (const url of tryUrls) {
    try {
      const r = await axios.post(url, { data: [prompt] }, { timeout: 120000 });
      // Many spaces return { data: [...] }
      if (r.data?.data && r.data.data.length) return String(r.data.data[0]);
      // Some return generated_text
      if (r.data?.generated_text) return String(r.data.generated_text);
      // else try next
    } catch (e) {
      // continue to next
    }
  }
  throw new Error('HF Space did not return valid output');
}

async function callHfApi(prompt, modelUrl = process.env.HF_URL) {
  if (!modelUrl || !process.env.HUGGINGFACE_API_KEY) throw new Error('HF API not configured');
  // modelUrl is like https://api-inference.huggingface.co/models/{model}
  const r = await axios.post(modelUrl, { inputs: prompt }, {
    headers: { Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}` },
    timeout: 120000
  });
  // Parse common HF shapes
  if (typeof r.data === 'string') return r.data;
  if (Array.isArray(r.data) && r.data[0]?.generated_text) return r.data[0].generated_text;
  if (r.data?.generated_text) return r.data.generated_text;
  if (r.data?.data && r.data.data[0]) return r.data.data[0];
  return JSON.stringify(r.data).slice(0, 4000);
}

/* ---------------------------
   Replicate generic call
   --------------------------- */
async function callReplicate(versionOrModel, input) {
  if (!process.env.REPLICATE_API_KEY) throw new Error('REPLICATE_API_KEY missing');
  const url = 'https://api.replicate.com/v1/predictions';
  const r = await axios.post(url, { version: versionOrModel, input }, {
    headers: {
      Authorization: `Token ${process.env.REPLICATE_API_KEY}`,
      'Content-Type': 'application/json'
    },
    timeout: 600000
  });
  return r.data;
}

/* ---------------------------
   Runway generic call
   --------------------------- */
async function callRunway(endpoint, model, input) {
  if (!process.env.RUNWAY_API_KEY) throw new Error('RUNWAY_API_KEY missing');
  const url = endpoint.replace(/\/$/, '');
  const r = await axios.post(url, { model, input }, {
    headers: {
      Authorization: `Bearer ${process.env.RUNWAY_API_KEY}`,
      'Content-Type': 'application/json'
    },
    timeout: 600000
  });
  return r.data;
}

/* ---------------------------
   ElevenLabs TTS
   --------------------------- */
async function callElevenLabsTTS(text) {
  if (!process.env.ELEVENLABS_API_KEY || !process.env.ELEVENLABS_VOICE_ID) throw new Error('ElevenLabs not configured');
  const url = `${process.env.ELEVENLABS_API_URL.replace(/\/$/, '')}/text-to-speech/${process.env.ELEVENLABS_VOICE_ID}`;
  const r = await axios.post(url, { text, voice_settings: {} }, {
    headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
    responseType: 'arraybuffer',
    timeout: 120000
  });
  return Buffer.from(r.data);
}

/* ---------------------------
   Web helpers: Wikipedia, DuckDuckGo, Google-translate (unofficial)
   --------------------------- */
async function wikiSummary(query) {
  try {
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(query)}`;
    const r = await axios.get(url, { timeout: 10000 });
    return r.data.extract || 'No summary found';
  } catch (e) {
    return 'Wikipedia lookup failed';
  }
}

async function duckDuck(query) {
  try {
    const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
    const r = await axios.get(url, { timeout: 8000 });
    if (r.data?.AbstractText) return r.data.AbstractText;
    const first = safeFirst(r.data?.RelatedTopics || []);
    if (first?.Text) return first.Text;
    return 'No DuckDuckGo instant answer';
  } catch (e) {
    return 'DuckDuckGo lookup failed';
  }
}

async function translateUnofficial(text, to = 'en') {
  try {
    const r = await translateLib(text, { to });
    return r.text;
  } catch (e) {
    return text;
  }
}

/* ---------------------------
   Model name -> HF model mapping (env-driven)
   --------------------------- */
const HF_MODEL_MAP = {
  llama2: process.env.MODEL_LLAMA2 || process.env.HF_URL || process.env.HF_MODEL,
  mistral: process.env.MODEL_MISTRAL || process.env.HF_URL,
  flan_t5: process.env.MODEL_FLAN_T5 || process.env.HF_URL,
  falcon: process.env.MODEL_FALCON || process.env.HF_URL,
  gpt2: process.env.MODEL_GPT2 || process.env.HF_URL,
  bloom: process.env.MODEL_BLOOM || process.env.HF_URL,
  default: process.env.MODEL || process.env.HF_URL || process.env.HF_MODEL
};

/* ---------------------------
   Unified /ai handler
   --------------------------- */
bot.command('ai', async (ctx) => {
  const text = ctx.message.text || '';
  const parts = text.trim().split(' ').slice(1);
  if (!parts.length) {
    return ctx.reply(withPrefix('Usage: /ai <mode> <input>\nModes: chat|wiki|duck|translate|media|tts|replicate'));
  }
  const mode = parts[0].toLowerCase();
  const rest = parts.slice(1).join(' ').trim();
  if (!rest && mode !== 'help') {
    return ctx.reply(withPrefix('Please provide input. Example: /ai chat Explain Newton laws'));
  }

  try {
    // HELP
    if (mode === 'help') {
      const help = `
/ai <mode> <input>

Chat (English default):
  /ai chat [model] <prompt>       -> Chat: model optional (llama2, mistral, flan_t5, falcon). Default llama2
Search:
  /ai wiki <topic>                -> Wikipedia summary
  /ai duck <query>                -> DuckDuckGo instant answer
Translate:
  /ai translate <text>            -> translate to English (use second arg 'am' to get Amharic)
Media (Runway / Replicate):
  /ai media <mode> <input>
    modes: t2i t2v i2v v2v upscale act flux fixface caption burncaption recon3d
TTS:
  /ai tts <text>                  -> ElevenLabs or Replicate TTS
Replicate chat:
  /ai replicate <modelname> <prompt> -> call replicate chat model as configured

Memory: conversation memory saved ~${MEMORY_TTL} seconds (if Redis configured)
      `;
      return ctx.reply(withPrefix(help));
    }

    // CHAT
    if (mode === 'chat') {
      // allow: /ai chat llama2 prompt...  OR /ai chat prompt...
      const firstToken = parts[1] ? parts[1].toLowerCase() : '';
      let modelKey = 'llama2';
      let prompt = rest;
      if (HF_MODEL_MAP[firstToken]) {
        modelKey = firstToken;
        prompt = parts.slice(2).join(' ');
      }
      if (!prompt) return ctx.reply(withPrefix('Please provide a prompt for chat.'));

      // memory
      const mem = await getMemory(ctx.from.id);
      mem.push({ role: 'user', content: prompt });
      // build context string (simple)
      const contextStr = mem.map((m) => `${m.role}: ${m.content}`).join('\n');

      // Try HF Space first (if configured)
      let answer = null;
      if (process.env.HF_SPACE_URL) {
        try {
          answer = await callHfSpace(contextStr, process.env.HF_SPACE_URL);
        } catch (e) {
          console.warn('HF Space failed:', e.message);
        }
      }
      // If no answer yet and HF API available for the requested model, call HF API
      if (!answer) {
        const hfModelUrl = HF_MODEL_MAP[modelKey] || HF_MODEL_MAP.default;
        if (hfModelUrl && process.env.HUGGINGFACE_API_KEY) {
          try {
            answer = await callHfApi(contextStr, hfModelUrl);
          } catch (e) {
            console.warn('HF API failed:', e.message);
          }
        }
      }
      // If still no answer, try Replicate chat model mapping envs
      if (!answer && process.env.REPLICATE_API_KEY) {
        try {
          // pick replicate chat model env if exists for this model key
          const repKeyName = ({
            llama2: 'REPLICATE_CHAT_MODEL_LLAMA2',
            gpt5: 'REPLICATE_CHAT_MODEL_GPT5',
            gpt4: 'REPLICATE_CHAT_MODEL_GPT4',
            gpt35: 'REPLICATE_CHAT_MODEL_GPT35',
            mistral: 'REPLICATE_CHAT_MODEL_MISTRAL'
          })[modelKey] || 'REPLICATE_CHAT_MODEL_GPT5';
          const repModel = process.env[repKeyName];
          if (repModel) {
            const r = await callReplicate(repModel, { prompt });
            // replicate may return outputs or urls — try to get text
            if (r.output && r.output.length) answer = String(r.output[0]);
            else if (r.status) answer = `Started replicate job (status ${r.status}). Check dashboard.`;
          }
        } catch (e) {
          console.warn('Replicate chat failed:', e.message);
        }
      }

      // fallback web-synthesis: wiki + duck
      if (!answer) {
        const wiki = await wikiSummary(prompt).catch(() => null);
        const ddg = await duckDuck(prompt).catch(() => null);
        answer = (wiki ? `From Wikipedia: ${wiki}` : '') + (ddg ? `\n\nDuck: ${ddg}` : '');
        if (!answer) answer = 'No provider available to answer this question.';
      }

      mem.push({ role: 'assistant', content: answer });
      await saveMemory(ctx.from.id, mem);
      return ctx.reply(withPrefix(answer));
    }

    // WIKI
    if (mode === 'wiki') {
      const summary = await wikiSummary(rest);
      return ctx.reply(withPrefix(summary));
    }

    // DUCK
    if (mode === 'duck') {
      const res = await duckDuck(rest);
      return ctx.reply(withPrefix(res));
    }

    // TRANSLATE
    if (mode === 'translate') {
      // allow: /ai translate am <text>  to translate to amharic
      const parts2 = rest.split(' ');
      let to = 'en';
      let textToTranslate = rest;
      if (parts2.length > 1 && parts2[0].length <= 3) {
        // treat first token as language code
        to = parts2[0];
        textToTranslate = parts2.slice(1).join(' ');
      }
      const t = await translateUnofficial(textToTranslate, to);
      return ctx.reply(withPrefix(t));
    }

    // TTS
    if (mode === 'tts') {
      // uses ElevenLabs if available, otherwise Replicate TTS if configured
      if (process.env.ELEVENLABS_API_KEY && process.env.ELEVENLABS_VOICE_ID) {
        try {
          const audioBuf = await callElevenLabsTTS(rest);
          return ctx.replyWithVoice({ source: audioBuf });
        } catch (e) {
          console.warn('ElevenLabs TTS failed:', e.message);
        }
      }
      if (process.env.REPLICATE_API_KEY && process.env.REPLICATE_TTS_MODEL) {
        try {
          const r = await callReplicate(process.env.REPLICATE_TTS_MODEL, { text: rest });
          const out = safeFirst(r.output);
          if (out) return ctx.replyWithVoice({ url: out });
        } catch (e) {
          console.warn('Replicate TTS failed:', e.message);
        }
      }
      return ctx.reply(withPrefix('No TTS provider configured.'));
    }

    // MEDIA unified (Runway + Replicate)
    if (mode === 'media') {
      // usage: /ai media <mode> <input>
      const sub = parts[1] ? parts[1].toLowerCase() : null;
      const payload = parts.slice(2).join(' ');
      if (!sub || !payload) return ctx.reply(withPrefix('Usage: /ai media <mode> <input>. Type /ai help for modes.'));
      // Runway modes
      const runwayModes = ['t2i', 't2v', 'i2v', 'v2v', 'upscale', 'act'];
      const repModes = ['flux', 'fixface', 'caption', 'burncaption', 'recon3d'];
      try {
        if (runwayModes.includes(sub) && process.env.RUNWAY_API_KEY) {
          let endpoint, model;
          switch (sub) {
            case 't2i': endpoint = process.env.RUNWAY_URL_TEXT_TO_IMAGE; model = process.env.RUNWAY_MODEL_TEXT_TO_IMAGE; break;
            case 't2v': endpoint = process.env.RUNWAY_URL_TEXT_TO_VIDEO; model = process.env.RUNWAY_MODEL_TEXT_TO_VIDEO; break;
            case 'i2v': endpoint = process.env.RUNWAY_URL_IMAGE_TO_VIDEO; model = process.env.RUNWAY_MODEL_IMAGE_TO_VIDEO; break;
            case 'v2v': endpoint = process.env.RUNWAY_URL_VIDEO_TO_VIDEO; model = process.env.RUNWAY_MODEL_VIDEO_TO_VIDEO; break;
            case 'upscale': endpoint = process.env.RUNWAY_URL_VIDEO_UPSCALE; model = process.env.RUNWAY_MODEL_VIDEO_UPSCALE; break;
            case 'act': endpoint = process.env.RUNWAY_URL_CHARACTER_PERFORMANCE; model = process.env.RUNWAY_MODEL_CHARACTER_PERFORMANCE; break;
          }
          const res = await callRunway(endpoint, model, (sub === 't2i' || sub === 't2v') ? { prompt: payload } : (sub === 'i2v' ? { image_url: payload } : { video_url: payload }));
          const out = safeFirst(res.output);
          if (!out) return ctx.reply(withPrefix('Runway returned no output.'));
          if (sub === 't2i' || sub === 'flux' || sub === 'fixface') return ctx.replyWithPhoto(out, { caption: withPrefix(payload) });
          return ctx.replyWithVideo(out, { caption: withPrefix(payload) });
        }
        // Replicate media
        if (repModes.includes(sub) && process.env.REPLICATE_API_KEY) {
          let repModel;
          switch (sub) {
            case 'flux': repModel = process.env.REPLICATE_IMAGE_MODEL; break;
            case 'fixface': repModel = process.env.REPLICATE_UPSCALE_MODEL; break;
            case 'caption': repModel = process.env.REPLICATE_VIDEO_CAPTION_MODEL; break;
            case 'burncaption': repModel = process.env.REPLICATE_VIDEO_CAPTIONED_MODEL; break;
            case 'recon3d': repModel = process.env.REPLICATE_3D_MODEL; break;
          }
          if (!repModel) return ctx.reply(withPrefix('Replicate model not set for this mode.'));
          const r = await callReplicate(repModel, (sub === 'flux') ? { prompt: payload } : (sub === 'fixface' ? { image: payload } : { video: payload }));
          const out = safeFirst(r.output);
          if (!out) return ctx.reply(withPrefix('Replicate returned no output.'));
          if (sub === 'flux' || sub === 'fixface') return ctx.replyWithPhoto(out, { caption: withPrefix(payload) });
          if (sub === 'recon3d') return ctx.replyWithDocument(out);
          if (sub === 'caption') return ctx.reply(withPrefix(out));
          return ctx.replyWithVideo(out, { caption: withPrefix(payload) });
        }
        return ctx.reply(withPrefix('No provider configured for that media mode.'));
      } catch (e) {
        return ctx.reply(withPrefix('Media error: ' + e.message));
      }
    }

    // Replicate direct chat: /ai replicate <modelkey> <prompt>
    if (mode === 'replicate') {
      const repKey = parts[1];
      const promptText = parts.slice(2).join(' ');
      if (!repKey || !promptText) return ctx.reply(withPrefix('Usage: /ai replicate <env_model_variable> <prompt>'));
      const repModel = process.env[repKey];
      if (!repModel) return ctx.reply(withPrefix(`No replicate model found in env as ${repKey}`));
      try {
        const r = await callReplicate(repModel, { prompt: promptText });
        const out = safeFirst(r.output) || JSON.stringify(r);
        return ctx.reply(withPrefix(out));
      } catch (e) {
        return ctx.reply(withPrefix('Replicate error: ' + e.message));
      }
    }

    // unknown mode
    return ctx.reply(withPrefix('Unknown mode. Type /ai help for usage.'));
  } catch (err) {
    console.error('AI handler error', err);
    return ctx.reply(withPrefix('Error: ' + (err && err.message ? err.message : String(err))));
  }
});

/* ---------------------------
   Start bot (webhook or polling)
   --------------------------- */
(async () => {
  try {
    // On Railway we prefer webhook; if BASE_URL provided, attempt setWebhook
    if (process.env.BASE_URL) {
      const url = `${process.env.BASE_URL.replace(/\/$/, '')}/webhook`;
      // try to set webhook (best-effort)
      try {
        const resp = await axios.get(`https://api.telegram.org/bot${BOT_TOKEN}/setWebhook?url=${encodeURIComponent(url)}`);
        console.log('setWebhook response:', resp.data);
      } catch (e) {
        console.warn('Failed to set webhook automatically:', e.message);
      }
      // start express to receive webhook (Telegraf can use webhook, but here we'll use polling to keep simple)
      // Many Railway setups accept polling. Use polling for simplicity.
      await bot.launch();
      console.log('Bot launched (polling).');
    } else {
      await bot.launch();
      console.log('Bot launched (polling).');
    }
  } catch (e) {
    console.error('Failed to launch bot:', e);
  }
})();
