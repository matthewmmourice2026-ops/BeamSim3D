#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;

uniform vec3 lightDir;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 viewPos;
uniform vec4 colDiffuse;
uniform float shininess;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 toLight = normalize(-lightDir);

    float diff = max(dot(normal, toLight), 0.0);

    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 halfwayDir = normalize(toLight + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    vec3 albedo = colDiffuse.rgb * fragColor.rgb;
    vec3 color = albedo * (ambientColor + lightColor * diff) + lightColor * spec * 0.35;

    finalColor = vec4(color, colDiffuse.a * fragColor.a);
}
