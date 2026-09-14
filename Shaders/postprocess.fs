#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;
uniform vec4 colDiffuse;

out vec4 finalColor;

void main() {
    vec4 texelColor = texture(texture0, fragTexCoord);

    vec2 uv = fragTexCoord - 0.5;
    float vignette = smoothstep(0.8, 0.3, length(uv));
    vec3 color = texelColor.rgb * mix(0.6, 1.0, vignette);

    // mild contrast + saturation boost
    color = (color - 0.5) * 1.08 + 0.5;
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, 1.15);

    finalColor = vec4(color, texelColor.a) * colDiffuse * fragColor;
}
