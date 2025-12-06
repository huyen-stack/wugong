import os
import json
from typing import Dict, Any, List

import streamlit as st

# -------------------------
# 1. Gemini 配置（记得设置环境变量 GEMINI_API_KEY）
# -------------------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

MODEL_NAME = "gemini-2.0-pro"  # 按你实际可用的模型改


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_AVAILABLE or not api_key:
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    return model


SYSTEM_PROMPT = """
你是一个专业的【动作电影分镜师 + AI 视频提示词写手】。

现在要用 Sora / Veo / Runway 这一类视频模型，生成 9:16 竖屏的武打短片。

你会收到一个 JSON 对象 spec_json，里面包含：
- clip_config：时长、画幅、整体风格标签
- characters：主角、对手，以及可能存在的其他角色 extras（如街头群殴的一打二、一打三）
- combo_plan：武打动作组合的简要说明（不要改变招式顺序）
- camera_plan：分镜和运镜意图（每个镜头的时间范围和重点）
- extra_controls：是否要微表情、环境反馈、血腥程度、安全限制等
- output_prefs：需要输出哪几种格式

你的任务是：

1）根据 spec_json，写出一段【英文 AI 视频提示词（prompt）】：
   - 面向 Sora / Veo / Runway 等视频模型。
   - 必须包含：
     - 场景与世界观（地点、时代、光线、整体气氛）
     - 主角与对手的外形和服装（保持前后一致），如有 extras 也要合理出现在画面中
     - 连贯的动作描述（预备 → 出招 → 命中 → 收招），严格按 combo_plan 的顺序
     - 被打者的物理反应（头部/躯干/腿的摆动、后退、踉跄），如是群殴要写清多人的互动
     - 运镜与镜头语言（每个 shot 的景别、机位、运动方向、是否手持）
     - 适当的环境反馈（灰尘、绳索、器械、地面等的反应），前提是不违背 extra_controls
   - 尽量用清晰、具体的描述，少用“awesome, cool, epic”等空洞形容词。
   - 遵守 safety_constraints，例如：如果 blood = "none"，就不要写见血画面。

2）然后，写出一段【中文时间轴分镜脚本】：
   - 按镜头输出，例如：
     【S01 | 0.0-0.5 秒】
     画面内容：...
     人物动作：...
     被打反应：...
     机位与运镜：...
     环境与细节：...
   - 必须覆盖 camera_plan 里的每个 shot，可以适度细化。
   - 如果有 extras（如第三个人参与群殴），要在分镜里交代清楚是谁在做什么。
   - 保持逻辑连贯，同一个人物的外形、服装和受伤状态前后一致。

3）输出格式：
   - 先输出英文视频提示词（用一段或多段英文）
   - 然后空一行，输出“—— 中文时间轴分镜 ——”
   - 再输出中文的分镜脚本。
   - 不要输出 JSON，不要解释你的思路。
"""


def call_gemini_with_spec(spec: Dict[str, Any]) -> str:
    model = get_gemini_client()
    if model is None:
        return (
            "【错误】Gemini 未配置成功，请确认：\n"
            "1）已安装 google-generativeai\n"
            "2）已设置 GEMINI_API_KEY 环境变量。"
        )

    spec_str = json.dumps(spec, ensure_ascii=False, indent=2)

    user_prompt = f"""
下面是本次视频片段的结构化规格说明 spec_json：

```json
{spec_str}
chat = model.start_chat(history=[
    {"role": "system", "parts": [SYSTEM_PROMPT]},
    {"role": "user", "parts": [user_prompt]}
])

resp = chat.send_message("请开始生成。")
return resp.text
-------------------------
2. 预设数据（角色 / 风格 / 动作套餐 / 运镜）
-------------------------

CHARACTERS: Dict[str, Dict[str, Any]] = {
"female_cn_sanda": {
"name": "女主 - 中国散打",
"role": "female_pro_fighter",
"nationality_style": "Chinese modern",
"visual_brief": (
"22-year-old athletic Chinese woman with a long black ponytail, "
"wearing a black sports bra, tight black training pants and black MMA gloves, "
"light sweat on her skin"
),
"motion_personality": "sharp_and_calm"
},
"male_us_mma": {
"name": "男对手 - 美国 MMA",
"role": "male_fighter",
"nationality_style": "US MMA",
"visual_brief": (
"stocky male fighter in his late 20s with short dark hair and slight beard stubble, "
"wearing red fight shorts and 4oz MMA gloves"
),
"motion_personality": "aggressive_but_tiring"
},
"male_hk_80s_thug": {
"name": "男对手 - 港片小混混",
"role": "male_thug",
"nationality_style": "Hong Kong 1980s",
"visual_brief": (
"lean Hong Kong street thug in a wrinkled white shirt, loose dark trousers and worn leather shoes, "
"slightly messy hair"
),
"motion_personality": "wild_and_showoff"
},
"female_wuxia_swordswoman": {
"name": "女主 - 武侠女侠",
"role": "female_swordswoman",
"nationality_style": "Chinese ancient wuxia",
"visual_brief": (
"elegant young swordswoman in flowing light-colored robes, "
"long black hair tied partly up, sword sheath on her back"
),
"motion_personality": "graceful_but_deadly"
},
"male_cn_street_punk": {
"name": "男角色 - 现代街头混混",
"role": "male_street_punk",
"nationality_style": "Chinese modern street",
"visual_brief": (
"young Chinese street punk in a dark hoodie, ripped jeans and sneakers, "
"short spiky hair, a bit cocky"
),
"motion_personality": "reckless_and_aggressive"
},
"male_cn_street_big": {
"name": "男角色 - 现代街头壮汉",
"role": "male_street_heavy",
"nationality_style": "Chinese modern street",
"visual_brief": (
"broad-shouldered Chinese man in a bomber jacket, dark pants and boots, "
"short hair, heavy build"
),
"motion_personality": "slow_but_powerful"
}
}

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
"cn_modern_sanda_gym": {
"label": "中国现代 - 散打馆",
"style_tags": ["cn_modern_sanda", "gym_interior", "cinematic"],
"description": "现代中国散打训练馆，冷色荧光灯、沙袋、擂台、镜面墙。"
},
"hk_80s_factory": {
"label": "香港 80s - 工厂/仓库",
"style_tags": ["hk_80s_kungfu", "warehouse", "stylized"],
"description": "80年代港片风格，老工厂或仓库，木箱、铁链、灰尘光束。"
},
"cn_wuxia_courtyard": {
"label": "古代武侠 - 山门/院落",
"style_tags": ["cn_wuxia", "ancient_courtyard", "fantasy_cinematic"],
"description": "古代武林门派山门或庭院，石板地、木柱、飘动的布幡和树叶。"
},
"us_mma_cage": {
"label": "美国 UFC 笼斗",
"style_tags": ["us_mma", "cage_arena", "sports_cinematic"],
"description": "MMA 笼子擂台，强烈顶光，周围观众在黑暗中吼叫。"
},
"cn_modern_street_night": {
"label": "中国现代 - 夜晚街头停车场",
"style_tags": ["cn_modern_street", "parking_lot_night", "gritty_cinematic"],
"description": "城市夜晚空旷停车场，路灯、霓虹反射在湿漉漉地面，适合街头群殴。"
}
}

COMBO_PRESETS: Dict[str, Dict[str, Any]] = {
"combo_jab_cross_lowkick": {
"label": "直拳 + 重拳 + 低扫",
"description": "a fast left jab to the face, a heavy right cross, then a powerful right low kick to the lead thigh",
"default_duration": 1.8
},
"combo_block_cross": {
"label": "格挡 + 右重拳反击",
"description": "she blocks an incoming strike, then fires a heavy right cross to the opponent's head",
"default_duration": 1.2
},
"combo_clinch_knee_push": {
"label": "抱颈 + 膝撞 + 推开",
"description": "she secures a clinch, drives a hard knee into the body, then shoves the opponent away",
"default_duration": 1.8
},
# 新增：武侠轻功版连招
"combo_wuxia_qinggong_sword": {
"label": "武侠轻功：闪身 + 拔剑 + 腾空一击",
"description": (
"she uses light-footwork to vanish from the opponent's line of attack, "
"appears at a new angle, draws her sword in one fluid motion, then launches into "
"a brief airborne slash before landing lightly on a stone railing"
),
"default_duration": 2.8
},
# 新增：街头群殴一打二
"combo_street_brawl_3p": {
"label": "街头群殴：一打二组合",
"description": (
"the main fighter faces two attackers at once: she elbows the attacker on her left, "
"then front-kicks the one on her right, before grabbing one of them and shoving him "
"hard into a parked car"
),
"default_duration": 2.5
}
}

CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
"dynamic_close": {
"label": "动态近景格斗风",
"shots_template": "jab_cross_lowkick",
"description": "手持近景 + 低机位跟腿 + 中景收尾。"
},
"wide_reveal": {
"label": "宽幅环境展示风",
"shots_template": "wide_focus",
"description": "开头环境大全景，中景打斗，最后拉远。"
},
# 新增：街头群殴用的运镜风格
"street_brawl_dynamic": {
"label": "街头群殴 - 混乱动态运镜",
"shots_template": "street_brawl_3p",
"description": "略带手持抖动的大景 + 中景切换，突出一打二的混乱感和被撞车辆等环境反馈。"
}
}

def build_camera_shots(template_name: str, duration_sec: float) -> List[Dict[str, Any]]:
"""根据简单模板和时长，生成 shots 列表。"""
if template_name == "jab_cross_lowkick":
# 按 3 段分：S01 出拳，S02 低扫，S03 被踢后退
t1 = round(duration_sec * 0.3, 2)
t2 = round(duration_sec * 0.8, 2)
t3 = round(duration_sec, 2)
return [
{
"shot_id": "S01",
"time_range": [0.0, t1],
"brief": "tight medium handheld shot framing both fighters from the waist up as she throws the jab and cross, camera slightly below eye level",
"priority": "show_face_and_gloves_impact"
},
{
"shot_id": "S02",
"time_range": [t1, t2],
"brief": "low tracking shot near the floor that follows the arc of her right shin slamming into his thigh, emphasizing muscle vibration and his leg buckling",
"priority": "show_kick_power_and_leg_reaction"
},
{
"shot_id": "S03",
"time_range": [t2, t3],
"brief": "medium shot pulling back slightly to show him stumbling sideways, catching his balance and revealing more of the environment",
"priority": "show_overall_reaction_and_space"
}
]
elif template_name == "wide_focus":
t1 = round(duration_sec * 0.25, 2)
t2 = round(duration_sec * 0.7, 2)
t3 = round(duration_sec, 2)
return [
{
"shot_id": "S01",
"time_range": [0.0, t1],
"brief": "wide establishing shot showing the whole space and both fighters circling each other",
"priority": "show_environment_and_positions"
},
{
"shot_id": "S02",
"time_range": [t1, t2],
"brief": "medium shot focusing on the main fighter as she lands the key strikes",
"priority": "show_main_actions"
},
{
"shot_id": "S03",
"time_range": [t2, t3],
"brief": "wide or medium-wide shot showing the aftermath and how both fighters are positioned after the exchange",
"priority": "show_aftermath"
}
]
elif template_name == "street_brawl_3p":
# 街头群殴一打二：S01 大景交代三人，S02 近景连续击打，S03 被撞到车上
t1 = round(duration_sec * 0.3, 2)
t2 = round(duration_sec * 0.75, 2)
t3 = round(duration_sec, 2)
return [
{
"shot_id": "S01",
"time_range": [0.0, t1],
"brief": "wide shot in a dimly lit parking lot at night, showing the main fighter facing two attackers, wet pavement reflecting neon lights",
"priority": "show_three_characters_and_environment"
},
{
"shot_id": "S02",
"time_range": [t1, t2],
"brief": "chaotic handheld medium shot that stays close as she elbows the attacker on the left and front-kicks the one on the right, camera reacting to each hit",
"priority": "show_elbow_and_front_kick_impacts"
},
{
"shot_id": "S03",
"time_range": [t2, t3],
"brief": "medium-wide shot as she grabs one attacker and shoves him hard into a parked car, the car shakes and the other attacker recovers in the background",
"priority": "show_shove_and_environment_reaction"
}
]
else:
# 默认单镜头
return [
{
"shot_id": "S01",
"time_range": [0.0, round(duration_sec, 2)],
"brief": "single continuous medium shot showing the whole exchange",
"priority": "show_whole_action"
}
]

-------------------------
3. 组装 spec_json 的函数
-------------------------

def build_spec_json(
duration_sec: float,
style_preset_key: str,
main_char_key: str,
opp_char_key: str,
extra_char_key: str,
combo_key: str,
energy_level: str,
violence_level: str,
camera_preset_key: str,
include_micro: bool,
include_breath: bool,
include_env: bool,
include_camera_detail: bool,
blood_level: str,
audio_hint: str
) -> Dict[str, Any]:

style_preset = STYLE_PRESETS[style_preset_key]
style_tags = style_preset["style_tags"]

main_char = CHARACTERS[main_char_key]
opp_char = CHARACTERS[opp_char_key]
combo = COMBO_PRESETS[combo_key]
camera_preset = CAMERA_PRESETS[camera_preset_key]

shots = build_camera_shots(camera_preset["shots_template"], duration_sec)

characters_block: Dict[str, Any] = {
    "main": {
        "id": "main_fighter",
        "role": main_char["role"],
        "nationality_style": main_char["nationality_style"],
        "visual_brief": main_char["visual_brief"],
        "motion_personality": main_char["motion_personality"]
    },
    "opponent": {
        "id": "opponent_fighter",
        "role": opp_char["role"],
        "nationality_style": opp_char["nationality_style"],
        "visual_brief": opp_char["visual_brief"],
        "motion_personality": opp_char["motion_personality"]
    }
}

# 如果选了第三角色（用于群殴/围攻），加入 extras
if extra_char_key != "none" and extra_char_key in CHARACTERS:
    extra_char = CHARACTERS[extra_char_key]
    characters_block["extras"] = [
        {
            "id": "extra_fighter_1",
            "role": extra_char["role"],
            "nationality_style": extra_char["nationality_style"],
            "visual_brief": extra_char["visual_brief"],
            "motion_personality": extra_char["motion_personality"]
        }
    ]

spec = {
    "clip_config": {
        "duration_sec": duration_sec,
        "aspect_ratio": "9:16",
        "style_tags": style_tags,
        "energy_level": energy_level,
        "violence_level": violence_level
    },
    "characters": characters_block,
    "combo_plan": {
        "combo_id": combo_key,
        "high_level_description": combo["description"],
        "tempo": "explosive_then_brief_pause",
        "intensity": "high"
    },
    "camera_plan": {
        "overall_style": camera_preset["label"],
        "shots": shots
    },
    "extra_controls": {
        "include_micro_expressions": include_micro,
        "include_breath_sweat_fatigue": include_breath,
        "include_environment_reaction": include_env,
        "include_camera_details": include_camera_detail,
        "blood": blood_level,
        "audio_hint": audio_hint,
        "safety_constraints": "no graphic gore, follow platform rules, respect the blood setting."
    },
    "output_prefs": {
        "need_english_video_prompt": True,
        "need_chinese_timeline": True,
        "timeline_step": 0.1
    }
}
return spec

-------------------------
4. Streamlit APP UI
-------------------------

st.set_page_config(page_title="武打分镜提示词工厂 PRO", layout="wide")

st.title("🥋 武打分镜提示词工厂 PRO（Gemini 版本）")

st.markdown(
"通过选择【角色 / 世界观 / 动作套餐 / 运镜风格 / 细节开关】，"
"自动生成结构化 spec_json 并调用 Gemini 输出：\n\n"
"- 英文视频提示词（给 Sora / Veo / Runway 等使用）\n"
"- 中文时间轴分镜脚本（方便你自己调教和复用）"
)

st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
st.header("① 基础设置")

style_key = st.selectbox(
    "世界观 / 风格预设",
    options=list(STYLE_PRESETS.keys()),
    format_func=lambda k: STYLE_PRESETS[k]["label"]
)
st.caption(STYLE_PRESETS[style_key]["description"])

main_char_key = st.selectbox(
    "主角角色",
    options=list(CHARACTERS.keys()),
    format_func=lambda k: CHARACTERS[k]["name"]
)
opp_char_key = st.selectbox(
    "对手角色",
    options=list(CHARACTERS.keys()),
    index=1 if len(CHARACTERS) > 1 else 0,
    format_func=lambda k: CHARACTERS[k]["name"]
)

# 第三角色（可选，用于群殴 / 围攻）
extra_options = ["none"] + list(CHARACTERS.keys())
extra_char_key = st.selectbox(
    "第三角色（可选，用于街头群殴 / 围攻场景）",
    options=extra_options,
    format_func=lambda k: "（无额外角色）" if k == "none" else CHARACTERS[k]["name"]
)

combo_key = st.selectbox(
    "动作套餐（连招）",
    options=list(COMBO_PRESETS.keys()),
    format_func=lambda k: COMBO_PRESETS[k]["label"]
)
default_dur = COMBO_PRESETS[combo_key]["default_duration"]
duration_sec = st.slider(
    "片段总时长（秒）",
    min_value=0.8,
    max_value=6.0,
    value=float(default_dur),
    step=0.1
)

energy_level = st.selectbox("能量强度（Energy Level）", ["low", "medium", "high"], index=2)
violence_level = st.selectbox("暴力程度（Violence Level）", ["soft", "moderate", "hard"], index=1)

camera_preset_key = st.selectbox(
    "运镜风格预设",
    options=list(CAMERA_PRESETS.keys()),
    format_func=lambda k: CAMERA_PRESETS[k]["label"]
)
st.caption(CAMERA_PRESETS[camera_preset_key]["description"])

st.header("② 细节与安全控制")

include_micro = st.checkbox("加入微表情 / 眼神细节", value=True)
include_breath = st.checkbox("加入呼吸 / 体力消耗状态", value=True)
include_env = st.checkbox("加入环境反馈（灰尘 / 道具 / 绳索等）", value=True)
include_camera_detail = st.checkbox("加入镜头焦段 / 景深 / 运动模糊等细节", value=True)

blood_level = st.selectbox("血腥程度 blood", ["none", "light", "visible"], index=0)
audio_hint = st.text_input(
    "声音 / 节奏提示（英文简写）",
    value="short sharp impact sounds, ambient noise, do not overdescribe music"
)

generate_btn = st.button("🚀 生成 spec_json 并调用 Gemini", type="primary")


with col_right:
st.header("③ 结果预览")

if generate_btn:
    spec_json = build_spec_json(
        duration_sec=duration_sec,
        style_preset_key=style_key,
        main_char_key=main_char_key,
        opp_char_key=opp_char_key,
        extra_char_key=extra_char_key,
        combo_key=combo_key,
        energy_level=energy_level,
        violence_level=violence_level,
        camera_preset_key=camera_preset_key,
        include_micro=include_micro,
        include_breath=include_breath,
        include_env=include_env,
        include_camera_detail=include_camera_detail,
        blood_level=blood_level,
        audio_hint=audio_hint
    )

    st.subheader("🔧 结构化 spec_json（可以下载 / 复用）")
    st.json(spec_json, expanded=False)

    st.download_button(
        "💾 下载 spec_json",
        data=json.dumps(spec_json, ensure_ascii=False, indent=2),
        file_name="spec_json_fight_clip.json",
        mime="application/json"
    )

    st.subheader("🧠 调用 Gemini 生成文案")
    with st.spinner("正在调用 Gemini 生成英文 Prompt + 中文时间轴分镜..."):
        text = call_gemini_with_spec(spec_json)

    # 尝试把英文和中文部分分开显示
    if "—— 中文时间轴分镜 ——" in text:
        en_part, zh_part = text.split("—— 中文时间轴分镜 ——", 1)
    else:
        en_part, zh_part = text, ""

    st.markdown("**① 英文视频提示词（给 Sora / Veo / Runway 用）**")
    st.text_area("English Prompt", en_part.strip(), height=260)

    if zh_part.strip():
        st.markdown("**② 中文时间轴分镜脚本**")
        st.text_area("中文分镜", zh_part.strip(), height=260)
    else:
        st.info("未能自动分割中英文，完整输出如下：")
        st.text_area("完整输出", text, height=400)
else:
    st.info("在左侧完成配置后，点击「🚀 生成 spec_json 并调用 Gemini」。")


---

### 接下来你可以直接这样玩：

- 想要**武侠轻功连招**：
  - 选 `世界观 = 古代武侠 - 山门/院落`
  - 主角选「女主 - 武侠女侠」
  - 动作套餐选「武侠轻功：闪身 + 拔剑 + 腾空一击」
  - 运镜随便选 `dynamic_close` 或 `wide_reveal`（以后可以再加“轻功专用运镜”）

- 想要**街头群殴一打二**：
  - 世界观选「中国现代 - 夜晚街头停车场」
  - 主角可以选「女主 - 中国散打」或其他
  - 对手选一个街头角色，比如「现代街头混混」
  - 第三角色选「现代街头壮汉」  
  - 动作套餐选「街头群殴：一打二组合」
  - 运镜选「街头群殴 - 混乱动态运镜」

Gemini 会根据你给的 `spec_json` 自动把**一打二 + 多角色分镜 + 环境反馈**写进英文 Prompt 和中文分镜里。

如果你后面还想加：

- 「武侠轻功空中对撞」  
- 「多人围殴镜头 + POV 第一视角」  
- 「自动拼接多段 Clip 成 15 秒短片」  

随时可以在这套框架上继续往上叠，我可以帮你一起扩展。
::contentReference[oaicite:0]{index=0}
