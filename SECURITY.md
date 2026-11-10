# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

이 도구는 네트워크 포트를 열지 않습니다. `hf:` 경로를 쓸 때만 허브에 HTTP Range 요청을 보내며, `HF_TOKEN` 은 리다이렉트 대상에 전달하지 않습니다. / This tool does not open any network port; it only sends HTTP Range requests to the hub when you use an `hf:` path, and `HF_TOKEN` is never forwarded to redirect targets.
