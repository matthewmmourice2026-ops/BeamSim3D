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
uniform vec3 fogColor;
uniform float fogStart;
uniform float fogEnd;
uniform sampler2D texture0;

out vec4 finalColor;

void main() {
    vec3 normal = normalize(fragNormal);
    vec3 toLight = normalize(-lightDir);

    float diff = max(dot(normal, toLight), 0.0);

    vec3 viewDir = normalize(viewPos - fragPosition);
    vec3 halfwayDir = normalize(toLight + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    // texture0 defaults to raylib's 1x1 white pixel for any model that
    // never gets an explicit diffuse texture (e.g. ThrowGame.cpp's terrain/
    // ball), so this multiply is a no-op there and only visibly tiles a
    // texture for meshes that assign one (race_game's ground/road).
    vec3 texel = texture(texture0, fragTexCoord).rgb;
    vec3 albedo = colDiffuse.rgb * fragColor.rgb * texel;
    vec3 color = albedo * (ambientColor + lightColor * diff) + lightColor * spec * 0.35;

    // Terrain now stretches out to x=3200 (see ThrowGame.cpp buildTerrainModel)
    // to fit max-power throws, so the far end needs to fade into the sky
    // instead of hard-cutting at the mesh edge.
    float dist = length(viewPos - fragPosition);
    float fogT = clamp((dist - fogStart) / max(fogEnd - fogStart, 0.001), 0.0, 1.0);
    color = mix(color, fogColor, fogT);

    finalColor = vec4(color, colDiffuse.a * fragColor.a);
}
